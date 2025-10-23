import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_experts: int = 2,
        num_langs: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_head = nn.Linear(hidden_dim, num_experts)
        self.lang_head = nn.Linear(hidden_dim, num_langs)

    def forward(self, embedding: torch.Tensor) -> dict:
        h = self.feature(embedding)
        expert_logits = self.gate_head(h)
        expert_weights = torch.softmax(expert_logits, dim=-1)
        lang_logits = self.lang_head(h)
        return {"expert_weights": expert_weights, "lang_logits": lang_logits}


