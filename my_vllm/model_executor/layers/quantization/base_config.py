"""Base classes for quantization configuration and methods."""

from abc import ABC, abstractmethod

import torch


class QuantizeMethodBase(ABC):
    """Abstract base for per-layer quantization methods.

    Each quantized layer (LinearBase, FusedMoE, Attention) receives
    a QuantizeMethodBase subclass that handles weight processing
    and forward pass dispatch.
    """

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create quantized weight parameters for a layer."""
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the quantized linear transform to input x."""
        raise NotImplementedError


class QuantizationConfig(ABC):
    """Abstract base for quantization configuration.

    Maps to a specific quant method (FP8, AWQ, GPTQ, etc.) and
    provides the get_quant_method() dispatch for each layer type.
    """

    packed_modules_mapping: dict[str, list[str]] = {}

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        raise NotImplementedError

    @abstractmethod
    def get_min_capability(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        """Return the QuantizeMethodBase for the given layer.

        Returns None if this layer should be skipped (unquantized).
        """
        raise NotImplementedError

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "QuantizationConfig":
        raise NotImplementedError

    @classmethod
    def get_from_keys(cls, config: dict, keys: list[str]) -> str:
        for key in keys:
            if key in config:
                return config[key]
        raise KeyError(f"None of {keys} found in config")

    @classmethod
    def get_from_keys_or(cls, config: dict, keys: list[str], default=None):
        try:
            return cls.get_from_keys(config, keys)
        except KeyError:
            return default
