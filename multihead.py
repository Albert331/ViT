import torch.nn as nn
import torch
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(64,64)
        self.k=nn.Linear(64,64)
        self.v=nn.Linear(64,64)

    def forward(self,x):
       q = self.q(x)
       k = self.k(x)
       v = self.v(x)

       wei =  q @ k.transpose(1,2)

       wei = wei/(64**0.5)
       wei = F.softmax(wei,dim=-1)

       out = wei @ v
       return out


class MultiHead(nn.Module):
    def __init__(self,num_heads):
        super().__init__()  
        self.heads =  nn.ModuleList([Head() for _ in range(num_heads)])  
        self.proj = nn.Linear(768, 768)
    def forward(self,x):
        chunks = x.chunk(len(self.heads), dim=-1)  
        
        outputs = []

        for head, chunk in zip(self.heads, chunks):
            outputs.append(head(chunk))

        out = torch.cat(outputs, dim=-1)
        out = self.proj(out)
        return out

          