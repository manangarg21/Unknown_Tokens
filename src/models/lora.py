from typing import Any


def attach_lora(model: Any, r: int = 8, alpha: int = 16, dropout: float = 0.1) -> Any:
    # Import lazily to avoid peft/transformers version import issues at module import time
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LoRA requested but 'peft' (and compatible 'transformers') is not available. "
            "Either disable LoRA in the config or install compatible versions, e.g.:\n"
            "pip install -U 'transformers>=4.34.0,<5' 'peft>=0.5.0'"
        ) from e

    # Using FEATURE_EXTRACTION as we wrap a bare AutoModel backbone (not *ForSequenceClassification)
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="FEATURE_EXTRACTION")
    return get_peft_model(model, config)
