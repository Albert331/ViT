import torch.nn as nn
import torch


from multihead import MultiHead
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.multiheadattn = MultiHead(12)
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(128)
    def forward(self,x):
       


        res1 = x
        x= self.ln1(x)
        x = self.multiheadattn(x)
        x=x+res1

        res2=x
        x=self.ln2(x)
        x = self.mlp(x)
        x=res2+x
        return x