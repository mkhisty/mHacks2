import pickle
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import os
import matplotlib.image as mpimg
import gzip
import torchvision.transforms.functional as TF
import zlib


class nblock(nn.Module):
    def __init__(self,c,encoder:bool,k, s,an ):
        self.helper = [c[0]]
        super(nblock, self).__init__()
        self.enc = encoder
        self.an = an
        self.c = c
        self.k = k
        self.s = s
        layers = []
        skips = []
        for i in range(0,len(k)):
            l4 = self.makel(i)
            layers.append(l4[0])
            if((i+1)%an==0):
                skips.append(l4[1])
        self.layers = nn.ModuleList(layers)
        self.skips = nn.ModuleList(skips)
    def makel(self,i):
        if (i+1)%self.an==0:
            rv =  (nn.Sequential(nn.ConvTranspose2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]),nn.LeakyReLU()),nn.Conv2d(self.helper[-1],self.c[i+1],1,1),) if self.enc ==False else (nn.Sequential(nn.Conv2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]),nn.LeakyReLU()),nn.Conv2d(self.helper[-1],self.c[i+1],1,1))
            self.helper.append(self.c[i+1])
        else:
            return(nn.Sequential(nn.ConvTranspose2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]))) if self.enc == False else nn.Sequential(nn.Conv2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]),)
        return rv

    def forward(self,x):
        x0=[x.clone()]
        y=x
#        for i in range(len(self.layers)):
#            y = self.layers[i][0](y)
#            if((i+1))%self.an==0:
#                y = torch.add(y,self.layers[i][1](transforms.Resize(y.shape[-2:])(x0)))
#                x0 = y
        n=0
        for i, l in enumerate(self.layers):
            y = l(y)
            if(i+1)%self.an==0:
                #y = torch.add(y,self.skips[n](transforms.Resize(y.shape[-2:])(x0[-1])))
                #x0.append(y.clone())
                n+=1
        return y       
class rblock(nn.Module):
    def __init__(self,c,encoder:bool,k, s,an ):
        self.helper = [c[0]]
        super(rblock, self).__init__()
        self.enc = encoder
        self.an = an
        self.c = c
        self.k = k
        self.s = s
        layers = []
        skips = []
        for i in range(0,len(k)):
            l4 = self.makel(i)
            layers.append(l4[0])
            if((i+1)%an==0):
                skips.append(l4[1])
        self.layers = nn.ModuleList(layers)
        self.skips = nn.ModuleList(skips)
    def makel(self,i):
        if (i+1)%self.an==0:
            rv =  (nn.Sequential(nn.ConvTranspose2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]),nn.LeakyReLU()),nn.Conv2d(self.helper[-1],self.c[i+1],1,1)) if self.enc ==False else (nn.Sequential(nn.Conv2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]),nn.LeakyReLU()),nn.Conv2d(self.helper[-1],self.c[i+1],1,1))
            self.helper.append(self.c[i+1])
        else:
            return(nn.Sequential(nn.ConvTranspose2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]))) if self.enc == False else nn.Sequential(nn.Conv2d(in_channels=self.c[i],out_channels= self.c[i+1] ,kernel_size= self.k[i],stride= self.s[i]))
        return rv

    def forward(self,x):
        x0=[x.clone()]
        y=x
#        for i in range(len(self.layers)):
#            y = self.layers[i][0](y)
#            if((i+1))%self.an==0:
#                y = torch.add(y,self.layers[i][1](transforms.Resize(y.shape[-2:])(x0)))
#                x0 = y
        n=0
        for i, l in enumerate(self.layers):
            y = l(y)
            if(i+1)%self.an==0:
                y = torch.add(y,self.skips[n](transforms.Resize(y.shape[-2:])(x0[-1])))
                x0.append(y.clone())
                n+=1
        return y
        

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net2 = nblock((3,9,12,15),True,(3,3,3),(2,2,2),2)
        self.net = rblock((15,12,9,3,3),False,(3,3,3,8),(2,2,2,1),2)
    def forward(self, x,compress):
        if(compress):
            x = self.net2(x)
            return x
        else:
            x = self.net(x)
            return x
model = Decoder()
device = torch.device('cpu')

model.load_state_dict(torch.load("fgen15.pt", map_location=device))


class dataset2(Dataset):
    def __init__(self):
        self.t = transforms.Compose([transforms.ToTensor(),])
        self.trainp = 0.9
    def __len__(self):
        return 20
    def __getitem__(self,i):
        img = self.t(mimg.imread("test_images/0006.png"))
        return img
d = dataset2()
#t = transforms.Compose([transforms.ToTensor(),transforms.CenterCrop((250,250)),transforms.Resize((150,150))])
img = mpimg.imread("test_images/0006.png")
img = TF.to_pil_image(img)
img.save("outputs/compressed.jpeg", "JPEG", quality=1)

loader = DataLoader(d,20)
testa = next(iter(loader))
testimg = testa[0]
testimg = torch.permute(testimg,(1,2,0))
compressed = model(torch.unsqueeze(testa[0].to(device),0),True)
reconstructed = torch.permute(model(compressed,False).detach().to("cpu"),(0,2,3,1))[0]

print(compressed.shape)
compressed=np.array(compressed.detach()[0][0])
print(compressed.shape)
torch.save(compressed,"outputs/bird3.pt")
c = compressed
#np.save("bird4.npy",c,allow_pickle=False)

def save(object, filename, protocol = 0):
        """Saves a compressed object to disk
        """
        file = gzip.GzipFile(filename, 'wb')
        file.write(pickle.dumps(object, protocol))
        file.close()
save(c,"outputs/bird10.pix")

torch.save(torch.unsqueeze(testa[0].to(device),0),"outputs/orig.pt")
f, axarr = plt.subplots(1,2) 
axarr[0].imshow(testimg)
axarr[1].imshow(reconstructed)



plt.show()
