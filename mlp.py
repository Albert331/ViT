import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128,512)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(512,128)

    def forward(self,x):
        x=self.fc1(x)
        x=self.gelu(x)
        x=self.fc2(x)
        return x    
