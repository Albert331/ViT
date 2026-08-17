import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import v2

from TransformerBlock import TransformerBlock

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






