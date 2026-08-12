"""LoRARequest — aligns with vllm/lora/request.py:8."""


class LoRARequest:
    """Request for a LoRA adapter.

    Mirrors vllm/lora/request.py:8 — LoRARequest(msgspec.Struct).

    lora_int_id must be globally unique (> 0) for a given adapter.
    """

    def __init__(
        self,
        lora_name: str,
        lora_int_id: int,
        lora_path: str = "",
        base_model_name: str | None = None,
    ) -> None:
        if lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {lora_int_id}")
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path
        self.base_model_name = base_model_name

    @property
    def adapter_id(self) -> int:
        return self.lora_int_id

    @property
    def name(self) -> str:
        return self.lora_name

    @property
    def path(self) -> str:
        return self.lora_path

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, self.__class__)
            and self.lora_name == value.lora_name
        )

    def __hash__(self) -> int:
        return hash(self.lora_name)

    def __repr__(self) -> str:
        return (
            f"LoRARequest(lora_name={self.lora_name!r}, "
            f"lora_int_id={self.lora_int_id})"
        )
