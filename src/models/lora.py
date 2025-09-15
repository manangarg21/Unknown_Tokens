from typing import Optional
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model


def attach_lora(model: PreTrainedModel, r: int = 8, alpha: int = 16, dropout: float = 0.1) -> PreTrainedModel:
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="SEQ_CLS")
    return get_peft_model(model, config)
