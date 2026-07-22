import torch
import torch.nn as nn

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, frames = x.shape

        y = self.pool(x).view(batch_size, channels)
        y = self.fc(y)
        y = y.unsqueeze(2)
        out = x * y

        return out




