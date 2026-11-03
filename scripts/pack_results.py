"""Zip the whole results directory, checkpoints included, with the standard library only."""

import argparse
import zipfile
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results_bundle.zip"))
    return parser.parse_args(argv)


def pack(results_dir: Path, output: Path) -> int:
    """Write every file under ``results_dir`` to ``output``; archive paths start with its name."""
    results_dir, output = results_dir.resolve(), output.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"{results_dir} is not a directory")
    if output.is_relative_to(results_dir):
        raise ValueError("the archive must be written outside the results directory")
    files = sorted(path for path in results_dir.rglob("*") if path.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(results_dir.parent).as_posix())
    temporary.replace(output)
    return len(files)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    count = pack(args.results_dir, args.output)
    print(f"{count} files -> {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
