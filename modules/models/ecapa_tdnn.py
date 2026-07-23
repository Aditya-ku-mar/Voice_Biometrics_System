import torch
import torch.nn as nn
import torch.nn.functional as F

from .se_res2block import SERes2Block
from .mfa import MultiLayerFeatureAggregation
from .attentive_pooling import AttentiveStatisticsPooling


class ECAPATDNN(nn.Module):

    def __init__(
        self,
        input_dim=80,
        channels=512,
        scale=8,
        emb_dim=192
    ):
        super().__init__()
        
        # Initial TDNN Layer
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            bias=False
        )

        self.bn = nn.BatchNorm1d(
            channels,
            eps=1e-5,
            momentum=0.1
        )

        self.relu = nn.ReLU(inplace=True)

        # SE-Res2 Blocks
        self.layer1 = SERes2Block(
            channels=channels,
            scale=scale,
            dilation=2
        )

        self.layer2 = SERes2Block(
            channels=channels,
            scale=scale,
            dilation=3
        )

        self.layer3 = SERes2Block(
            channels=channels,
            scale=scale,
            dilation=4
        )

        # Multi-Layer Feature Aggregation
        self.mfa = MultiLayerFeatureAggregation(
            channels=channels,
            num_blocks=3
        )

        # Attentive Statistics Pooling
        self.pool = AttentiveStatisticsPooling(
            channels=channels * 3
        )

        self.bn_pool = nn.BatchNorm1d(
            channels * 6
        )

        # Embedding Layer
        self.embedding = nn.Sequential(
            nn.Linear(
                channels * 6,
                emb_dim
            ),
            nn.BatchNorm1d(
                emb_dim
            )
        )

    def forward(self, x):

        # Initial TDNN
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # SE-Res2 Blocks
        out1 = self.layer1(x)
        out2 = self.layer2(
            x + out1
        )
        out3 = self.layer3(
            x + out1 + out2
        )

        # Multi-Layer Feature Aggregation
        x = self.mfa(
            [out1, out2, out3]
        )
        
        # Attentive Statistics Pooling
        x = self.pool(x)
        x = self.bn_pool(x)

        # Speaker Embedding
        embedding = self.embedding(x)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding