from typing import Tuple
import torch
import torch.nn as nn


class RCNNHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        conv_channels: int = 256,
        kernel_sizes=(2, 3, 4),
        rnn_hidden: int = 256,
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden_size, out_channels=conv_channels, kernel_size=k) for k in kernel_sizes]
        )
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        total_features = len(kernel_sizes) * conv_channels + 2 * rnn_hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_features, num_labels)

    def forward(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # sequence_output: [B, T, H]
        x = sequence_output.transpose(1, 2)  # [B, H, T]
        conv_feats = []
        for conv in self.convs:
            c = torch.relu(conv(x))  # [B, C, T']
            c = torch.max(c, dim=-1).values  # [B, C]
            conv_feats.append(c)
        rnn_out, _ = self.rnn(sequence_output)  # [B, T, 2*H]
        rnn_pooled = (rnn_out * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        feats = torch.cat(conv_feats + [rnn_pooled], dim=-1)
        logits = self.classifier(self.dropout(feats))
        return logits
