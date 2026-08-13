"""ApplyRotaryEmb — aligns with vLLM vllm/model_executor/layers/rotary_embedding/common.py.

Architecture preserved: ApplyRotaryEmb (CustomOp), forward_static (neox/gptj styles).

Engineering details skipped: CUDA kernel dispatch, flashinfer, flash_attn, HIP.
"""

import torch

from my_vllm.model_executor.custom_op import CustomOp


class ApplyRotaryEmb(CustomOp):
    """Apply rotary embedding to query/key tensors — aligns with vLLM ApplyRotaryEmb.

    Args:
        is_neox_style: Whether to use Neox-style (default) or GPT-J-style.
        enable_fp32_compute: Temporarily compute in FP32 for accuracy.
    """

    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        super().__init__(enforce_enable=enforce_enable)
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch(opt), seq_len, num_heads, head_size]
            cos: [seq_len, rotary_dim // 2]
            sin: [seq_len, rotary_dim // 2]
            is_neox_style: Neox-style (default) or GPT-J-style.
        """
        origin_dtype = x.dtype
        if enable_fp32_compute:
            x = x.float()

        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        if is_neox_style:
            output = torch.cat((o1, o2), dim=-1)
        else:
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        if enable_fp32_compute:
            output = output.to(origin_dtype)
        return output

    def forward_native(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_static(
            x, cos, sin, self.is_neox_style, self.enable_fp32_compute
        )

    def extra_repr(self) -> str:
        s = f"is_neox_style={self.is_neox_style}"
        s += f", enable_fp32_compute={self.enable_fp32_compute}"
        return s
