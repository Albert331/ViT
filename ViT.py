import torch.nn as nn
import torch
import torch.nn.functional as F


class ViT(nn.module):
    def __init__(self):
        super().__init__()