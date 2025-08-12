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
import math

def getimgblock(arr, idx, partrow, partcol):
    band, r, c = arr.shape
    # rnum = r / partrow
    # cnum = c / partcol
    rnum = math.ceil(r / partrow)
    cnum = math.ceil(c / partcol)   
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
            'w':self.w
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
    # annfile = 'E:/datasets/AfriSAR2016/lopenp_TM140_16008_002_160225_L090HH_03_BC.ann'

    # datafile = 'E:/datasets/kapok_output/kapokfile111.hdf5'

    # outpath = 'E:/datasets/kapok_output/mycnn/'
    # logpath =r"E:\codes\TreeHeightDL\kapok-main\scripts\mylogs"


    annfile = '/home/hailiang/datasets/UAVSAR/Lope/lopenp_TM140_16008_002_160225_L090HH_03_BC.ann'
    datafile = '/home/hailiang/datasets/UAVSAR/lope_kapok.h5'
    outpath = '/home/hailiang/codes/SAR2/output/lope/PIDLF/'
    logpath ='/home/hailiang/codes/SAR2/output/lope/PIDLF/logs/'

    # First index is the azimuth size, second index is the range size.

    mlwin = [20,5]

    throd = 60



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
    scene.opt()

    #Now, we create a mask which identifies low HV backscatter areas.
    mask = scene.power('HV') # Get the HV backscattered power (in linear units).
    mask[mask <= 0] = 1e-10 # Get rid of zero-valued power.
    mask = 10*np.log10(mask) # Convert to dB.
    mask = mask > -22 # Find pixels above/below -22 dB threshold.

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
    # return
    coh = np.array(coh)
    coh[coh>throd] = throd

    dem = np.array(dem)

    dem[dem>throd] = throd

    patchsize = 30
    dataset = MyDataset(hh,hv,vv,coh,dem, patchrow= patchsize, patchcol= patchsize)

    # test dataset
    # tem = dataset.__getitem__(15)
    # num = dataset.__len__()
    # showimg(tem["x"][0])

# ---------------------------------------------------------


#    start_time = time.process_time()
    # print("begin time.....: %s" % (start_time))

    # on local computer

    checkpoints_name ="v1-lope-000.pth"
    times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logpath = os.path.join(logpath,str(times))
    print("   logpath is : %s"%logpath)
    checkpoint_path = os.path.join(logpath, checkpoints_name)

    batch_size = 64
    channels = 6
    epochs = 100000
    seed = 42
    savemodel_frequence = 10000
    logfrequence = 100
    sgdlr = 0.0001
    momentem = 0.9

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)


    # a = get_a()
    # print("paramater a is: %s ....." %a)
 
    # print("train data len:...",traindata.__len__())
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # criterion = Loss() #单独一个标签
    # n = int(a*100)
    # grid search
    # minvalloss = 1000

    model = SingleModel(channels, outchannel=1).to(device)

    para=[model, device, epochs, logfrequence, patchsize, batch_size,datafile,sgdlr]
    print(para)


    # print(model)
    # if torch.cuda.device_count() > 1:
    #     # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    #     # model = torch.nn.DataParallel(model, device_ids=[1])
    #     model = torch.nn.DataParallel(model)

    criterion = Loss() #单独一个标签
    # criterion = LossMix(lamda=a)# 综合两类标签的混合加权损失

    optimizer = optim.SGD(model.parameters(), lr=sgdlr,momentum=momentem,weight_decay=0.000001)
    optimizer = PCGrad(optimizer)



        # print("train loader len:...",len(train_loader))
        # writer= SummaryWriter(logpath)

    for epoch in range(0,epochs):
        model.train()
        # if (epoch+1)%30000 == 0:
        #     sgdlr = sgdlr/10
        #     optimizer = optim.SGD(model.parameters(), lr=sgdlr, momentum=momentem, weight_decay=0.000001)
        totalloss = 0.

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x = data["x"].to(device).to(torch.float32)
            y1 = data["coh"].to(device).to(torch.float32) # coh
            y2 = data["dem"].to(device).to(torch.float32) # DEM
            # y1 = y1.unsqueeze(1)
            # y2 = y2.unsqueeze(1)
            out= model(x)
        
            loss1 = criterion(out,y1)
            loss2 = criterion(out,y2)

            optimizer.pc_backward([loss1,loss2])
            # sum(losses).backward()
            optimizer.step()
        
            totalloss =totalloss + loss1 + loss2
        if (epoch + 1) % logfrequence== 0:
            times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            print("    %s:epoch %s/%s-avgtra-loss:%s" % (times, (epoch+1), epochs, str(totalloss / (len(train_loader)))))
            # writer.add_scalar('avg-trainloss', totalloss / (len(train_loader)), global_step=epoch)

        if (epoch + 1) % savemodel_frequence == 0:

            if not os.path.exists(logpath):
                os.makedirs(logpath)
            pathname = str(epoch + 1) + "epoch.pth"

            # on local computer
            torch.save(model.state_dict(), os.path.join(logpath, pathname))
            
            # on kaggle
            # torch.save(model.state_dict(), ("./"+pathname))

            times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            print("    %s:save %s successfully" % (times, pathname))

    
    # fished_time = time.process_time()
    # program_run_time= fished_time - start_time
  
    # print("finished time.....: %s" % (fished_time))
    # print("    run time.....: %s" % (program_run_time))

    print("finished...............")



if __name__ == "__main__":
    main()














