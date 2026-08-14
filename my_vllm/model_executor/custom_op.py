"""CustomOp — platform-aware kernel dispatch abstraction.

Real vLLM architecture (vllm/model_executor/custom_op.py):

  CustomOp (nn.Module):
    - Base class for platform-specific kernel operations
    - Dispatches forward() to forward_cuda / forward_hip / forward_native
      based on current_platform
    - Registered in global op_registry dict for OOT (out-of-tree) replacement
    - Supports OOT override via op_registry_oot

  PluggableLayer (nn.Module):
    - Module-composing abstraction (instantiates sub-modules)
    - No per-platform forward dispatch — entire class is OOT-replaceable
    - Used for attention, MoE, and other complex layers

  op_registry / op_registry_oot:
    - Global dicts mapping name → CustomOp/PluggableLayer subclass
    - OOT registrations override in-tree implementations at instantiation time
"""

import torch
import torch.nn as nn

from my_vllm.platforms import current_platform


op_registry: dict[str, type] = {}
op_registry_oot: dict[str, type] = {}


def maybe_get_oot_by_class(class_type: type) -> type:
    class_name = class_type.__name__
    if class_name in op_registry_oot:
        return op_registry_oot[class_name]
    return class_type


def _resolve_forward_method(op: "CustomOp") -> callable:
    """Pick the platform-correct forward_* method.

    Real vLLM uses ``current_platform.is_cuda()`` / ``is_rocm()`` /
    ``is_xpu()`` to choose among ``forward_cuda`` / ``forward_hip``
    / ``forward_xpu``.  We mirror that mapping so subclasses that
    implement only the platform-specific variants are routed
    correctly at call time.
    """
    if current_platform.is_cuda():
        method = getattr(op, "forward_cuda", None)
        if method is not None and getattr(method, "__code__", None) is not getattr(
            CustomOp.forward_cuda, "__code__", None
        ):
            return method
    elif current_platform.is_rocm():
        method = getattr(op, "forward_hip", None)
        if method is not None and getattr(method, "__code__", None) is not getattr(
            CustomOp.forward_hip, "__code__", None
        ):
            return method
    elif current_platform.is_xpu():
        method = getattr(op, "forward_xpu", None)
        if method is not None and getattr(method, "__code__", None) is not getattr(
            CustomOp.forward_xpu, "__code__", None
        ):
            return method
    elif current_platform.is_cpu():
        method = getattr(op, "forward_cpu", None)
        if method is not None and getattr(method, "__code__", None) is not getattr(
            CustomOp.forward_cpu, "__code__", None
        ):
            return method
    return op.forward_native


class PluggableLayer(nn.Module):
    """
    Base class for pluggable layers.

    A PluggableLayer is a *module-composing* abstraction: it may instantiate
    other nn.Module objects as sub-layers. Supports out-of-tree (OOT)
    replacement of the entire layer class at instantiation time.
    """

    def __new__(cls, *args, **kwargs):
        try:
            layer_class_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@PluggableLayer.register."
            ) from None

        if layer_class_name not in op_registry_oot:
            layer_cls_to_instantiate = cls
        else:
            layer_cls_to_instantiate = op_registry_oot[layer_class_name]
        return super().__new__(layer_cls_to_instantiate)

    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_registry[name] = op_cls
            return op_cls
        return decorator

    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):
        def decorator(layer_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in op_registry_oot, f"Duplicate layer name: {reg_name}"
            layer_cls.name = reg_name
            op_registry_oot[reg_name] = layer_cls
            return layer_cls

        if _decorated_layer_cls is None:
            return decorator
        elif isinstance(_decorated_layer_cls, type):
            return decorator(_decorated_layer_cls)
        else:
            raise TypeError("Decorator can only be applied to classes.")


class CustomOp(nn.Module):
    """
    Base class for custom ops with platform-aware dispatch.

    Dispatches forward() to:
      - forward_cuda()  — NVIDIA CUDA kernels
      - forward_hip()   — AMD ROCm kernels
      - forward_native() — PyTorch native (fallback / torch.compile)

    Registered in op_registry for OOT replacement.
    """

    def __new__(cls, *args, **kwargs):
        try:
            op_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@CustomOp.register."
            ) from None

        if op_name not in op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = op_registry_oot[op_name]
        return super().__new__(op_cls_to_instantiate)

    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        super().__init__()
        self._enforce_enable = enforce_enable
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def dispatch_forward(self, compile_native: bool = False):
        """Select the platform-correct forward_* method.

        Mirrors the vLLM ``CustomOp.dispatch_forward`` (which keys
        on ``current_platform``).  The default returns
        ``forward_native`` so subclasses that don't implement a
        platform-specific variant fall back to the PyTorch path.
        """
        if compile_native:
            return self.forward_native
        return _resolve_forward_method(self)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation (optional). Used for torch.compile or testing."""
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def enabled(cls) -> bool:
        """Check if this op is enabled on the current platform."""
        return True

    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_registry[name] = op_cls
            return op_cls
        return decorator

    @classmethod
    def register_oot(cls, name: str):
        def decorator(op_cls):
            assert name not in op_registry_oot, f"Duplicate OOT op name: {name}"
            op_cls.name = name
            op_registry_oot[name] = op_cls
            return op_cls
        return decorator
