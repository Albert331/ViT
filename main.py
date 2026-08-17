import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import v2
import warnings
warnings.filterwarnings('ignore')
from Vit import Vit




data_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])





if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model =  Vit()
    model = model.to(device)

    fullES =torchvision.datasets.EuroSAT(root='./data',download=True,transform = data_transform)


    train,test = random_split(fullES,[int(0.8 * len(fullES)),len(fullES) - (int(0.8 * len(fullES)))])


    train_loader  = DataLoader(train,shuffle=True,batch_size=32,num_workers=8,persistent_workers=True,pin_memory=True)
    test_loader  = DataLoader(test,shuffle=True,batch_size=32,num_workers=8,persistent_workers=True,pin_memory=True)




    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    epoch = 15
    gradScaler = torch.amp.GradScaler(device=device)

    print('='*50)
    for _ in range(epoch):
        model.train()
        losslist=[]
        correct=0
        total=0
        for img,label in train_loader:
            img=img.to(device)
            label=label.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                out = model(img)
                loss = crit(out,label)

            gradScaler.scale(loss).backward()
            gradScaler.step(optimizer)
            gradScaler.update()

            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            losslist.append(loss.item())
        print(f'epoch:{_+1} | avg loss:{sum(losslist)/len(losslist)} | accuracy:{correct/total}')


        val_losslist=[]
        val_correct=0
        val_total=0
        model.eval()
        with torch.no_grad():
            for img,label in test_loader:
                img=img.to(device)
                label=label.to(device)
                out = model(img)
                loss = crit(out,label)
                
                pred = torch.argmax(out, dim=1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)
                val_losslist.append(loss.item())

            print(f'epoch:{_+1} | avg loss:{sum(val_losslist)/len(val_losslist)} | accuracy:{val_correct/val_total}')    
        print('='*50)