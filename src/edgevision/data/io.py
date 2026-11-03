"""Dataset contract checks, batch processing with a per-sequence cache, and output writing."""

import hashlib
import json
import zipfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.config import ExperimentConfig, PreprocessConfig
from edgevision.data.landmarks import num_selected
from edgevision.data.preprocess import PreprocessResult, SequenceInfo, preprocess_sequence
from edgevision.utils.runinfo import collect_run_info

Float16Array = npt.NDArray[np.float16]
Uint8Array = npt.NDArray[np.uint8]

TRAIN_CSV = "train.csv"
SIGN_MAP_JSON = "sign_to_prediction_index_map.json"
TRAIN_COLUMNS = ("path", "participant_id", "sequence_id", "sign")
SEQUENCE_COLUMNS = ("frame", "row_id", "type", "landmark_index", "x", "y", "z")
MANIFEST_COLUMNS = (
    "sequence_id",
    "participant_id",
    "sign",
    "label",
    "excluded",
    "reason",
    "t_raw",
    "t_kept",
    "mirrored",
    "norm_method",
)

LABEL_DTYPE = np.int16
PARTICIPANT_DTYPE = np.int32
SEQUENCE_DTYPE = np.int64

PROGRESS_EVERY = 1000
CHUNK_SIZE = 16
# Bump when a transform changes meaning, so old cache entries stop matching.
CACHE_VERSION = 1


def read_train_csv(path: Path) -> pd.DataFrame:
    """Read train.csv; NA parsing stays off so a sign spelled like "null" remains a string."""
    df = pd.read_csv(path, keep_default_na=False)
    if set(df.columns) != set(TRAIN_COLUMNS):
        raise ValueError(f"{path}: expected columns {TRAIN_COLUMNS}, got {list(df.columns)}")
    if df.empty:
        raise ValueError(f"{path}: no rows")
    for column in ("participant_id", "sequence_id"):
        if not pd.api.types.is_integer_dtype(df[column]):
            raise ValueError(f"{path}: column {column} must be integer, got {df[column].dtype}")
    for column in ("path", "sign"):
        if (df[column].astype(str) == "").any():
            raise ValueError(f"{path}: column {column} has empty values")
    duplicates = df.loc[df["sequence_id"].duplicated(), "sequence_id"]
    if not duplicates.empty:
        raise ValueError(f"{path}: duplicate sequence_id values: {duplicates.head(10).tolist()}")
    return df


def read_sign_map(path: Path) -> dict[str, int]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON object")
    if not all(type(index) is int for index in raw.values()):
        raise ValueError(f"{path}: every sign must map to an integer index")
    if sorted(raw.values()) != list(range(len(raw))):
        raise ValueError(f"{path}: indices must be exactly 0..{len(raw) - 1}")
    sign_map: dict[str, int] = raw
    return sign_map


def attach_labels(train_df: pd.DataFrame, sign_map: dict[str, int]) -> pd.DataFrame:
    unknown = sorted(set(train_df["sign"].astype(str)) - set(sign_map))
    if unknown:
        raise ValueError(f"signs missing from the sign map: {unknown[:10]}")
    return train_df.assign(label=train_df["sign"].map(sign_map).astype("int64"))


def _check_fits(values: "pd.Series[int]", dtype: type[np.signedinteger[Any]], name: str) -> None:
    limits = np.iinfo(dtype)
    low, high = int(values.min()), int(values.max())
    if low < limits.min or high > limits.max:
        raise OverflowError(f"{name} range [{low}, {high}] does not fit {limits.dtype}")


def load_table(cfg: PreprocessConfig) -> pd.DataFrame:
    """Validate train.csv and the sign map, then apply ``limit``.

    Runs before any sequence is processed so a contract violation fails at once.
    """
    table = attach_labels(
        read_train_csv(cfg.data_root / TRAIN_CSV), read_sign_map(cfg.data_root / SIGN_MAP_JSON)
    )
    _check_fits(table["participant_id"], PARTICIPANT_DTYPE, "participant_id")
    _check_fits(table["label"], LABEL_DTYPE, "label")
    return table if cfg.limit is None else table.head(cfg.limit)


def read_sequence(path: Path) -> pd.DataFrame:
    """Read one sequence parquet and validate its columns and dtypes.

    Pool workers already run in parallel, so pyarrow's own thread pool stays off.
    """
    df = pd.read_parquet(path, engine="pyarrow", use_threads=False)
    if set(df.columns) != set(SEQUENCE_COLUMNS):
        raise ValueError(f"expected columns {SEQUENCE_COLUMNS}, got {list(df.columns)}")
    for column in ("frame", "landmark_index"):
        if not pd.api.types.is_integer_dtype(df[column]):
            raise ValueError(f"column {column} must be integer, got {df[column].dtype}")
    for column in ("row_id", "type"):
        if not pd.api.types.is_string_dtype(df[column]):
            raise ValueError(f"column {column} must be string, got {df[column].dtype}")
    for column in ("x", "y", "z"):
        if not pd.api.types.is_float_dtype(df[column]):
            raise ValueError(f"column {column} must be float, got {df[column].dtype}")
    return df


def cache_fingerprint(cfg: PreprocessConfig) -> str:
    """Hash everything a cached sequence depends on, so changed settings miss the cache."""
    payload = {
        "version": CACHE_VERSION,
        "data_root": str(cfg.data_root),
        "num_frames": cfg.num_frames,
        "use_z": cfg.use_z,
        "pose_indices": cfg.pose_indices,
        "min_hand_frames": cfg.min_hand_frames,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class WorkerParams:
    data_root: Path
    cache_dir: Path | None
    num_frames: int
    use_z: bool
    pose_indices: tuple[int, ...]
    min_hand_frames: int


@dataclass(frozen=True)
class SequenceOutcome:
    """Storage-ready result for one sequence; arrays are None when it was excluded."""

    sequence_id: int
    features: Float16Array | None
    presence: Uint8Array | None
    info: SequenceInfo
    from_cache: bool = False


def to_storage(sequence_id: int, result: PreprocessResult) -> SequenceOutcome:
    """Narrow a result to on-disk dtypes.

    float16 tops out at 65504. A scale just above MIN_SCALE can exceed that and would become
    inf on the cast, so such a sequence is excluded rather than stored non-finite.
    """
    if result.features is None or result.hand_presence is None:
        return SequenceOutcome(sequence_id, None, None, result.info)
    with np.errstate(over="ignore"):
        features = result.features.astype(np.float16)
    if not np.isfinite(features).all():
        info = SequenceInfo(**{**result.info, "excluded": True, "reason": "float16_overflow"})
        return SequenceOutcome(sequence_id, None, None, info)
    return SequenceOutcome(
        sequence_id, features, result.hand_presence.astype(np.uint8), result.info
    )


def _compute(params: WorkerParams, sequence_id: int, relative_path: str) -> SequenceOutcome:
    try:
        df = read_sequence(params.data_root / relative_path)
        result = preprocess_sequence(
            df,
            num_frames=params.num_frames,
            use_z=params.use_z,
            pose_indices=params.pose_indices,
            min_hand_frames=params.min_hand_frames,
        )
    except Exception as error:  # One bad file is a manifest row, not a failed run.
        info = SequenceInfo(
            t_raw=0,
            t_kept=0,
            mirrored=False,
            norm_method=None,
            excluded=True,
            reason=f"read_error: {type(error).__name__}",
        )
        return SequenceOutcome(sequence_id, None, None, info)
    return to_storage(sequence_id, result)


def _load_cached(path: Path, sequence_id: int) -> SequenceOutcome | None:
    """Load a cached outcome; a missing or unreadable entry counts as a cache miss."""
    try:
        with np.load(path, allow_pickle=False) as entry:
            info: SequenceInfo = json.loads(str(entry["info"]))
            if "features" not in entry.files:
                return SequenceOutcome(sequence_id, None, None, info, from_cache=True)
            return SequenceOutcome(
                sequence_id, entry["features"], entry["presence"], info, from_cache=True
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _write_cached(path: Path, outcome: SequenceOutcome) -> None:
    arrays: dict[str, npt.NDArray[Any]] = {"info": np.array(json.dumps(outcome.info))}
    if outcome.features is not None and outcome.presence is not None:
        arrays["features"] = outcome.features
        arrays["presence"] = outcome.presence
    _save_npz(path, arrays)


def _save_npz(path: Path, arrays: dict[str, npt.NDArray[Any]]) -> None:
    """Write through a temporary name so an interrupted run never leaves a truncated file."""
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    temporary.replace(path)


def process_row(params: WorkerParams, row: tuple[int, str]) -> SequenceOutcome:
    """Top-level so it pickles under spawn; caches every outcome, exclusions included."""
    sequence_id, relative_path = row
    if params.cache_dir is None:
        return _compute(params, sequence_id, relative_path)
    cache_file = params.cache_dir / f"{sequence_id}.npz"
    cached = _load_cached(cache_file, sequence_id)
    if cached is not None:
        return cached
    outcome = _compute(params, sequence_id, relative_path)
    _write_cached(cache_file, outcome)
    return outcome


def _outcomes(
    params: WorkerParams, rows: Sequence[tuple[int, str]], num_workers: int
) -> Iterator[SequenceOutcome]:
    worker = partial(process_row, params)
    if num_workers == 1:
        yield from map(worker, rows)
        return
    # Spawn everywhere: forking after pyarrow has started threads can deadlock on Linux.
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
        yield from pool.map(worker, rows, chunksize=CHUNK_SIZE)


@dataclass(frozen=True)
class BatchResult:
    """Feature rows align with the manifest's non-excluded rows, in order."""

    manifest: pd.DataFrame
    features: Float16Array
    presence: Uint8Array
    cache_hits: int
    computed: int


def process_dataset(cfg: PreprocessConfig, *, use_cache: bool = True) -> BatchResult:
    """Process every row of the table, in parallel, keeping train.csv order in the outputs."""
    table = load_table(cfg)
    total = len(table)

    cache_dir = cfg.output_dir / "cache" / cache_fingerprint(cfg) if use_cache else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    params = WorkerParams(
        data_root=cfg.data_root,
        cache_dir=cache_dir,
        num_frames=cfg.num_frames,
        use_z=cfg.use_z,
        pose_indices=tuple(cfg.pose_indices),
        min_hand_frames=cfg.min_hand_frames,
    )

    shape = (total, cfg.num_frames, num_selected(cfg.pose_indices), 3 if cfg.use_z else 2)
    features: Float16Array = np.empty(shape, dtype=np.float16)
    presence: Uint8Array = np.empty((total, cfg.num_frames, 2), dtype=np.uint8)

    rows = list(zip(table["sequence_id"].tolist(), table["path"].tolist(), strict=True))
    infos: list[SequenceInfo] = []
    kept = cache_hits = 0
    for outcome in _outcomes(params, rows, cfg.num_workers):
        infos.append(outcome.info)
        cache_hits += outcome.from_cache
        if outcome.features is not None and outcome.presence is not None:
            features[kept] = outcome.features
            presence[kept] = outcome.presence
            kept += 1
        if len(infos) % PROGRESS_EVERY == 0:
            print(f"processed {len(infos)}/{total}", flush=True)

    return BatchResult(
        manifest=_build_manifest(table, infos),
        features=features[:kept],
        presence=presence[:kept],
        cache_hits=cache_hits,
        computed=total - cache_hits,
    )


def _build_manifest(table: pd.DataFrame, infos: Sequence[SequenceInfo]) -> pd.DataFrame:
    details = pd.DataFrame.from_records(infos, index=table.index)
    manifest = pd.concat([table, details], axis=1)
    manifest["reason"] = manifest["reason"].fillna("")
    manifest["norm_method"] = manifest["norm_method"].fillna("")
    return manifest.loc[:, list(MANIFEST_COLUMNS)].reset_index(drop=True)


def write_features(path: Path, batch: BatchResult) -> None:
    kept = batch.manifest.loc[~batch.manifest["excluded"]]
    arrays: dict[str, npt.NDArray[Any]] = {
        "X": batch.features,
        "presence": batch.presence,
        "y": kept["label"].to_numpy(dtype=LABEL_DTYPE),
        "participant_id": kept["participant_id"].to_numpy(dtype=PARTICIPANT_DTYPE),
        "sequence_id": kept["sequence_id"].to_numpy(dtype=SEQUENCE_DTYPE),
    }
    _save_npz(path, arrays)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_report(
    batch: BatchResult, config: ExperimentConfig, *, wall_time_s: float, use_cache: bool
) -> dict[str, Any]:
    manifest = batch.manifest
    kept = manifest.loc[~manifest["excluded"]]
    n_kept = len(kept)

    per_sign = kept.groupby("sign").size().reindex(manifest["sign"].unique(), fill_value=0)
    methods = kept["norm_method"].value_counts()
    reasons = manifest.loc[manifest["excluded"], "reason"].value_counts()
    return {
        "counts": {
            "total": len(manifest),
            "kept": n_kept,
            "excluded": len(manifest) - n_kept,
            "excluded_by_reason": {str(key): int(value) for key, value in reasons.items()},
        },
        "per_sign_kept": {
            "min": int(per_sign.min()),
            "median": float(per_sign.median()),
            "max": int(per_sign.max()),
        },
        "mirrored_fraction": float(kept["mirrored"].mean()) if n_kept else 0.0,
        "norm_method_fractions": {str(key): int(value) / n_kept for key, value in methods.items()},
        "cache": {"enabled": use_cache, "hits": batch.cache_hits, "computed": batch.computed},
        "wall_time_s": round(wall_time_s, 3),
        "config": config.model_dump(mode="json"),
        "run_info": collect_run_info(),
    }


def format_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    per_sign = report["per_sign_kept"]
    cache = report["cache"]
    lines = [
        f"sequences: total={counts['total']} kept={counts['kept']} excluded={counts['excluded']}",
        *(f"  excluded[{key}]={value}" for key, value in counts["excluded_by_reason"].items()),
        f"per-sign kept: min={per_sign['min']} median={per_sign['median']} max={per_sign['max']}",
        f"mirrored fraction: {report['mirrored_fraction']:.3f}",
        "norm methods: "
        + " ".join(f"{key}={value:.3f}" for key, value in report["norm_method_fractions"].items()),
        f"cache: enabled={cache['enabled']} hits={cache['hits']} computed={cache['computed']}",
        f"wall time: {report['wall_time_s']:.1f} s",
    ]
    return "\n".join(lines)
