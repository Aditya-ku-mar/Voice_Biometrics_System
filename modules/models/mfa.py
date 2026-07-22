import torch
import torch.nn as nn


class MultiLayerFeatureAggregation(nn.Module):

    def __init__(self, channels=512, num_blocks=3):
        super().__init__()
        self.out_channels = channels * num_blocks
        self.fusion = nn.Sequential(
            nn.Conv1d(
                self.out_channels,
                self.out_channels,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.out_channels)

        )

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fusion(x)

        return x