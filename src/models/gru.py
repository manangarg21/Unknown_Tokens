import torch
import torch.nn as nn


class GRUHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, hidden: int = 256, layers: int = 1, bidirectional: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_features = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_features, num_labels)

    def forward(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.gru(sequence_output)
        pooled = (rnn_out * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        logits = self.classifier(self.dropout(pooled))
        return logits
