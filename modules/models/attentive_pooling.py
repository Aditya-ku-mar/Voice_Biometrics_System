import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels=1536, bottleneck=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(
                in_channels=channels * 3,
                out_channels=bottleneck,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(
                in_channels=bottleneck,
                out_channels=channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        """
        x : (B, C, T)
        """

        B, C, T = x.size()

        # ----------------------------------------------------
        # Global Mean
        # Shape: (B, C, 1)
        # ----------------------------------------------------
        mean = torch.mean(
            x,
            dim=2,
            keepdim=True
        )

        # ----------------------------------------------------
        # Global Standard Deviation
        # Shape: (B, C, 1)
        # ----------------------------------------------------
        std = torch.sqrt(
            torch.var(
                x,
                dim=2,
                unbiased=False,
                keepdim=True
            ) + 1e-9
        )

        # ----------------------------------------------------
        # Broadcast mean & std to every frame
        # (B,C,1) -> (B,C,T)
        # ----------------------------------------------------
        mean = mean.expand(-1, -1, T)
        std = std.expand(-1, -1, T)

        # ----------------------------------------------------
        # Context Features
        # Concatenate:
        # Original Features
        # Global Mean
        # Global Std
        #
        # Shape:
        # (B,3C,T)
        # ----------------------------------------------------
        context = torch.cat(
            [x, mean, std],
            dim=1
        )

        # ----------------------------------------------------
        # Attention Scores
        # Shape:
        # (B,C,T)
        # ----------------------------------------------------
        alpha = self.attention(context)

        # Normalize across time
        alpha = F.softmax(alpha, dim=2)

        # ----------------------------------------------------
        # Weighted Mean
        # Shape:
        # (B,C)
        # ----------------------------------------------------
        mean = torch.sum(
            alpha * x,
            dim=2
        )

        # ----------------------------------------------------
        # Weighted Standard Deviation
        # ----------------------------------------------------
        second_moment = torch.sum(
            alpha * (x ** 2),
            dim=2
        )

        std = torch.sqrt(
            (second_moment - mean ** 2).clamp(min=1e-9)
        )

        # ----------------------------------------------------
        # Statistics Pooling
        #
        # Output:
        # (B,2C)
        # ----------------------------------------------------
        pooled = torch.cat(
            [mean, std],
            dim=1
        )

        return pooled