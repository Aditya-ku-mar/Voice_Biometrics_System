import torch
import torch.nn as nn
from models.res2net import Res2NetBlock
from models.se_block import SEBlock

class SERes2Block(nn.Module):
    def __init__(self,channels=512,scale=8,dilation=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

        self.res2net = Res2NetBlock(
            channels=channels,
            scale=scale,
            dilation=dilation

        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=1
            ),
            nn.BatchNorm1d(channels)
        )

        self.se = SEBlock(
            channels,
            reduction=8
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        identity = x
        out = self.conv1(x)
        out = self.res2net(out)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        out = self.relu(out)

        return out