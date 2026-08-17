import torch.nn as nn
import torch
from patch_embed import PatchEmbedding
from TransformerBlock import TransformerBlock

class Vit(nn.Module):
    def __init__(self):
        super().__init__()
        self.patching = PatchEmbedding()
        self.blocks=nn.ModuleList([TransformerBlock() for _ in range(6)])
                
        self.cls = nn.Parameter(torch.randn(1, 1, 128))
        self.pe = nn.Parameter(torch.randn(1, 197, 128))

    def forward(self,x):
        x = self.patching(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pe    
        for block in self.blocks:
            x = block(x)

        return x