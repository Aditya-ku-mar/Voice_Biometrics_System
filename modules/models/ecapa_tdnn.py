import torch
import torch.nn as nn
import torch.nn.functional as F

from models.se_res2block import SERes2Block
from models.mfa import MultiLayerFeatureAggregation
from models.attentive_pooling import AttentiveStatisticsPooling


class ECAPATDNN(nn.Module):

    def __init__(
        self,
        input_dim=80,
        channels=512,
        emb_dim=192
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

        self.layer1 = SERes2Block(channels=channels, dilation=2)
        self.layer2 = SERes2Block(channels=channels, dilation=3)
        self.layer3 = SERes2Block(channels=channels, dilation=4)

        self.mfa = MultiLayerFeatureAggregation(
            channels=channels,
            num_blocks=3
        )

        self.pool = AttentiveStatisticsPooling(
            channels=channels * 3
        )

        self.bn_pool = nn.BatchNorm1d(channels * 6)

        self.embedding = nn.Sequential(
            nn.Linear(channels * 6, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x):

        x = self.conv(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        x = self.mfa([out1, out2, out3])

        x = self.pool(x)
        x = self.bn_pool(x)

        embedding = self.embedding(x)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding


