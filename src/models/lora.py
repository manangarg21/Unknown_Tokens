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
    # For some architectures (e.g., DeBERTa), explicitly specifying target modules greatly improves reliability.
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""

    target_modules = None
    mt = model_type.lower()
    if mt in ("deberta", "deberta-v2", "deberta_v2", "deberta-v3"):
        # DeBERTa Attention uses query_proj/key_proj/value_proj/o_proj
        target_modules = ["query_proj", "key_proj", "value_proj", "o_proj"]
    elif mt in ("bert", "roberta", "xlm-roberta", "distilbert"):
        # Common BERT-like attention projections
        target_modules = ["query", "key", "value", "dense"]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=target_modules,
    )
    return get_peft_model(model, config)
