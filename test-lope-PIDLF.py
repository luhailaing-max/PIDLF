# -*- coding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import numpy as np
import os
import os.path
import kapok
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler

from torch.nn import functional as F
from pcgrad import PCGrad
import time

import random
import torch.optim as optim
from torch import  nn, einsum



def getimgblock(arr, idx, partrow, partcol):
    band, r, c = arr.shape
    rnum = r / partrow
    cnum = c / partcol
    tem = idx
    idr = int(tem // cnum)
    idc = int(tem % cnum)
    idrstart = partrow * idr
    idrend = partrow * idr + partrow
    idcstart = partcol * idc
    idcend = partcol * idc + partcol

    img= arr[:, idrstart:idrend, idcstart:idcend]
    return img


def padding(arr, partrow, partcol):
    band, r, c = arr.shape
    # print("padding before %s"%str(arr.shape))
    if r % partrow == 0:
        row = r
    else:
        row = r + (partrow - r % partrow)
    if c % partcol == 0:
        col = c
    else:
        col = c + (partcol - c % partcol)
    rowp = row - r
    colp = col - c
    arr = np.pad(arr, ((0, 0), (0, rowp), (0, colp)), "constant")
    # print("padding after %s"%str(arr.shape))
    return arr

class MyDataset(Dataset):
    def __init__(self, hh, hv, vv, dem, coh, patchrow, patchcol):
  
        hh1 = hh.real
        hh2 = hh.imag
        hv1 = hv.real
        hv2 = hv.imag
        vv1 = vv.real
        vv2 = vv.imag

        hh1 = torch.tensor(hh1).unsqueeze(0)
        hv1 = torch.tensor(hv1).unsqueeze(0)
        vv1 = torch.tensor(vv1).unsqueeze(0)

        hh2 = torch.tensor(hh2).unsqueeze(0)
        hv2 = torch.tensor(hv2).unsqueeze(0)
        vv2 = torch.tensor(vv2).unsqueeze(0)

        x = torch.cat((hh1,hh2,hv1,hv2,vv1,vv2), 0)
        # showimg(x[0])

        dem = torch.tensor(dem).unsqueeze(0)
        coh = torch.tensor(coh).unsqueeze(0)
        b, self.h, self.w = coh.shape

        self.x = padding(x,patchrow,patchcol)
        self.dem = padding(dem,patchrow,patchcol)
        self.coh = padding(coh,patchrow,patchcol)

        _, self.pah, self.paw = self.coh.shape
        self.patchrow = patchrow
        self.patchcol = patchcol
      
        # showimg(self.x[0])

    def __len__(self):
        band, r, c = self.x.shape
        rnum = r / self.patchrow
        cnum = c / self.patchcol
        num = int(rnum * cnum)

        return num

    def __getitem__(self, idx):
        # print("idx:%s"%idx)
        # hh = getimgblock(self.hh,idx,self.patchrow,self.patchcol)
        # hv = getimgblock(self.hv,idx,self.patchrow,self.patchcol)
        # vv = getimgblock(self.vv,idx,self.patchrow,self.patchcol)
        x = getimgblock(self.x,idx,self.patchrow,self.patchcol)
        dem = getimgblock(self.dem,idx,self.patchrow,self.patchcol)
        coh = getimgblock(self.coh,idx,self.patchrow,self.patchcol)
    
        output = {
            "idx":idx,
            "x": x,
            "dem": dem,
            "coh": coh,
            'patchsize': self.patchcol,
            'h': self.h,
            'w':self.w,
            'pah':self.pah,
            'paw': self.paw,
    
                  }
        return output

def showimg(arr):
    plt.figure(' arr')
    plt.imshow(arr)
    plt.colorbar(shrink = 0.8)
    plt.show()
    print("finished....")


class SingleModel(nn.Module):
    def __init__(self,inchannel, outchannel):
        super(SingleModel, self).__init__()
        self.con1 = nn.Conv2d(inchannel, 32, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.con2 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.con3 = nn.Conv2d(16,outchannel, kernel_size=5, padding=2)
        self.out = nn.Identity()

    def forward(self,x):
        x = self.con1(x)
        x = self.relu1(x)
        x = self.con2(x)
        x = self.relu2(x)
        x = self.con3(x)
        x = self.out(x)
        return x

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    def forward(self,x,y):
        loss = F.mse_loss(x,y)
        return loss

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    import kapok
    annfile = '/home/hailiang/datasets/UAVSAR/Lope/lopenp_TM140_16008_002_160225_L090HH_03_BC.ann'

    datafile = '/home/hailiang/datasets/UAVSAR/lope_kapok.h5'

    outpath = '/home/hailiang/codes/SAR2/output/lope/'
    logpath ='/home/hailiang/codes/SAR2/output/lope/PIDLF/logs'

    checkpoints_name ="10000epoch.pth"
    times = "202402081312"
    logpath = os.path.join(logpath,times)
    print("    logpath is : %s"%logpath)
    checkpoint_path = os.path.join(logpath, checkpoints_name)

    patchsize = 30
    throd = 100
    site = 'lope'
    method = 'PIDLF'

    # annfile = '/home/hllu/datasets'

    # datafile = '/home/hllu/datasets/kapok_output/kapokfile111.hdf5'

    # outpath = '/home/hllu/datasets/kapok_output/mycnn/'


    # logpath ="/home/hllu/codes/kapok-main/scripts/mylogs"

    # First index is the azimuth size, second index is the range size.
    mlwin = [20,5]

    if not os.path.exists(outpath):
        os.makedirs(outpath)


    if os.path.isfile(datafile):
        scene = kapok.Scene(datafile)
    else:
        import kapok.uavsar
        scene = kapok.uavsar.load(annfile,datafile,mlwin=mlwin)

        # scene = kapok.uavsar.load(
        #     'E:/datasets/AfriSAR2016/lopenp_TM140_16008_002_160225_L090HH_03_BC.ann',
        #     'E:/datasets/kapok_output/kapokfile.hdf5',
        #     tracks = [0,1], azbounds = [1000,3000],
        #     rngbounds = [1000,3000]
        # )
    # scene.opt()

    #Now, we create a mask which identifies low HV backscatter areas.
    # mask = scene.power('HV') # Get the HV backscattered power (in linear units).
    # mask[mask <= 0] = 1e-10 # Get rid of zero-valued power.
    # mask = 10*np.log10(mask) # Convert to dB.
    # mask = mask > -22 # Find pixels above/below -22 dB threshold.

    #only run onece.
    # tem1coh = scene.inv('sinc', name='sinc', bl=0, desc='sinc, hv and ext. free parameters, no temporal decorrelation.', mask=mask, overwrite=True)
    # tem2dem = scene.inv('dem', name='dem', bl=0, desc='sinc, hv and ext. free parameters, no temporal decorrelation.', mask=mask, overwrite=True)
    def getdata(scene,bl):
        hh = scene.coh('HH', bl=bl)
        hv = scene.coh('HV', bl=bl)
        vv = scene.coh('VV', bl=bl)
        coh = scene.get('sinc/hv')
        dem = scene.get('dem/hv')
        return hh, hv, vv, coh, dem



    hh, hv, vv, coh, dem = getdata(scene, bl=0)
    print("getdata finished.....")
    
    coh = np.array(coh)
    coh[coh>throd] = throd

    dem = np.array(dem)

    dem[dem>throd] = throd



    dataset = MyDataset(hh,hv,vv,coh,dem, patchrow= patchsize, patchcol= patchsize)

    # test dataset
    # tem = dataset.__getitem__(15)
    # num = dataset.__len__()
    # showimg(tem["x"][0])

# ---------------------------------------------------------


#    start_time = time.process_time()
    # print("begin time.....: %s" % (start_time))

    # on local computer




    batch_size = 4

    channels = 6
    # print("in channels is :",channels)


    epochs = 10000
    seed = 42
   
    savemodel_frequence = 10000
    sgdlr = 0.0001
    momentem = 0.9

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)


    # a = get_a()
    # print("paramater a is: %s ....." %a)
 
    # print("train data len:...",traindata.__len__())
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # criterion = Loss() #单独一个标签
    # n = int(a*100)
    # grid search
    # minvalloss = 1000

    model = SingleModel(channels, outchannel=1).to(device)# v1



    if os.path.exists(checkpoint_path):
        # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
        # model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path, map_location=torch.device(device)).items()})
        print("    Success to loading model dict from %s ....."%checkpoint_path)
    else:
        print("    Failed to load model dict  from %s ....."%checkpoint_path)
        return

    temdata = dataset.__getitem__(0)
    realr = temdata['h']
    realc = temdata['w']
    row = temdata['pah']
    col = temdata['paw']
    img = torch.zeros((1,row,col))
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(train_loader):
            idxx = data["idx"].to(device)

            x = data["x"].to(device).to(torch.float32)
            out= model(x)
            f2 = torch.squeeze(out)
            for i, f2i in enumerate(f2):
                rnum = row / patchsize
                cnum = col / patchsize
                #banchsize>1时使用以下语句
                tem = idxx[i]
                # tem = idxx
                idr = int(tem // cnum)
                idc = int(tem % cnum)
                idrstart = patchsize * idr
                idrend = patchsize * idr + patchsize
                idcstart = patchsize * idc
                idcend = patchsize * idc + patchsize
        
                #  无重叠或者重叠区域直接覆盖
                img[:, idrstart:idrend, idcstart:idcend] = f2i

    out = img[:, 0:realr, 0:realc]
    out = torch.squeeze(out).numpy()

    scene.geo(out,outpath+site+'_'+method+'.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

    print("finished...............")



if __name__ == "__main__":
    main()
































