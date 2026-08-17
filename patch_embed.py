
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3,128,8,stride=8)

    def forward(self,x):
        x=self.patch_embed(x)
        x=x.flatten(2)
        x=x.transpose(1,2)    
        return x
