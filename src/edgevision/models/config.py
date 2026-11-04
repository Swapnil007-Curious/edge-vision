"""Model hyperparameters; ``name`` selects the architecture when a config is parsed from YAML."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from edgevision.config import _StrictModel

ModelName = Literal["gru", "cnn", "transformer", "hybrid"]
# The v1 architectures: the v1 grid, quantisation and benchmarks iterate over these.
MODEL_NAMES: tuple[ModelName, ...] = ("gru", "cnn", "transformer")
ALL_MODEL_NAMES: tuple[ModelName, ...] = (*MODEL_NAMES, "hybrid")


class GRUConfig(_StrictModel):
    name: Literal["gru"] = "gru"
    proj_dim: int = Field(default=192, ge=1)
    hidden_size: int = Field(default=128, ge=1)
    num_layers: int = Field(default=2, ge=1)
    input_dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    rnn_dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)


class CNNConfig(_StrictModel):
    name: Literal["cnn"] = "cnn"
    channels: int = Field(default=192, ge=1)
    num_blocks: int = Field(default=6, ge=1)
    kernel_size: int = Field(default=5, ge=1)
    block_dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_kernel(self) -> Self:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so 'same' padding keeps the length")
        return self


class TransformerConfig(_StrictModel):
    name: Literal["transformer"] = "transformer"
    d_model: int = Field(default=192, ge=1)
    num_layers: int = Field(default=4, ge=1)
    num_heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=384, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_heads(self) -> Self:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        return self


class HybridConfig(_StrictModel):
    """CNN-transformer hybrid: ``num_stages`` x (``conv_blocks`` conv blocks + one transformer
    block). The head dropout is "late": 0.0 until ``late_dropout_start`` of the epochs are done."""

    name: Literal["hybrid"] = "hybrid"
    dim: int = Field(default=256, ge=1)
    num_stages: int = Field(default=3, ge=1)
    conv_blocks: int = Field(default=2, ge=1)
    kernel_size: int = Field(default=17, ge=1)
    expansion_dim: int = Field(default=512, ge=1)
    conv_dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    num_heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=512, ge=1)
    attention_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.4, ge=0.0, lt=1.0)
    late_dropout_start: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_shapes(self) -> Self:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so 'same' padding keeps the length")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        return self


AnyModelConfig = GRUConfig | CNNConfig | TransformerConfig | HybridConfig
ModelConfig = Annotated[AnyModelConfig, Field(discriminator="name")]


def default_model_config(name: ModelName) -> AnyModelConfig:
    if name == "gru":
        return GRUConfig()
    if name == "cnn":
        return CNNConfig()
    if name == "hybrid":
        return HybridConfig()
    return TransformerConfig()
