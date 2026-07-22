import torch
import torch.nn as nn

class Res2NetBlock(nn.Module):
    def __init__(self,channels,scale=8,kernel_size=3,dilation=2):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        
        self.convs = nn.ModuleList()
        for _ in range(scale - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.width,
                        self.width,
                        kernel_size,
                        padding=dilation,
                        dilation=dilation
                    ),
                    nn.BatchNorm1d(self.width),
                    nn.ReLU(inplace=True)
                )
            )
            
    def forward(self, x):
        splits = torch.chunk(
            x,
            self.scale,
            dim=1
        )
        outputs = [splits[0]]
        for i in range(1, self.scale):
            if i == 1:
                y = self.convs[i-1](splits[i])
            else:
                y = self.convs[i-1](
                    splits[i] + outputs[-1]
                )
            outputs.append(y)
        return torch.cat(outputs, dim=1)




