import torch
import torchvision
# from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import v2

data_transform = v2.Compose([
    v2.ToTensor(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

fullES =torchvision.datasets.EuroSAT(root='./data',download=True,transform = data_transform)

train,test = random_split(fullES,[int(0.8 * len(fullES)),len(fullES) - (int(0.8 * len(fullES)))])


train_loader  = DataLoader(train,shuffle=True,batch_size=32,num_workers=6,persistent_workers=True,pin_memory=True)
test_loader  = DataLoader(test,shuffle=True,batch_size=32,num_workers=6,persistent_workers=True,pin_memory=True)

fir = next(iter(train_loader))
print(fir)


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3,128,8,stride=8)

    def forward(self,x):
        x=self.patch_embed(x)
        x=x.flatten(2)
        x=x.transpose(1,2)    
        return x


class transformer_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(128)
        
