import torch
import torch.nn as nn


class AttentiveStatisticsPooling(nn.Module):

    def __init__(self, channels=1536, attention_channels=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(
                in_channels=channels * 3,
                out_channels=attention_channels,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels,
                out_channels=channels,
                kernel_size=1
            ),
        )

    def forward(self, x):

        T = x.size(-1)

        # Global mean
        mean = torch.mean(
            x,
            dim=2,
            keepdim=True
        )

        # Global standard deviation
        std = torch.sqrt(
            torch.var(
                x,
                dim=2,
                keepdim=True,
                unbiased=False
            ).clamp(min=1e-9)
        )

        # Repeat along time dimension
        mean = mean.repeat(1, 1, T)
        std = std.repeat(1, 1, T)

        # Context
        context = torch.cat(
            (
                x,
                mean,
                std
            ),
            dim=1
        )

        # Channel-dependent attention
        attention = self.attention(context)
        attention = torch.softmax(
            attention,
            dim=2
        )

        # Weighted mean
        mean = torch.sum(
            attention * x,
            dim=2
        )

        # Weighted standard deviation
        std = torch.sqrt(
            (
                torch.sum(
                    attention * (x ** 2),
                    dim=2
                ) - mean ** 2
            ).clamp(min=1e-9)
        )

        # Statistics pooling
        pooled = torch.cat(
            (
                mean,
                std
            ),
            dim=1
        )

        return pooled