#coding:utf8
# =============================================================================
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class SegDataset(Dataset):
    def __init__(self,filetxt,imagesize,cropsize,transform=None):
        lines = open(filetxt,'r').readlines()
        self.samples = []
        self.imagesize = imagesize
        self.cropsize = cropsize
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                   transforms.RandomHorizontalFlip(),
                   transforms.RandomRotation(15),
                   transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ])

        for line in lines:
            line = line.strip()
            imagepath,labelpath = line.split(' ')
            self.samples.append((imagepath,labelpath))
    ##每次读取一组图片
    def __getitem__(self,index):##获取数据
        imagepath,labelpath = self.samples[index]
        image = cv2.imread(imagepath)
        label = cv2.imread(labelpath,0) ##读成1通道
       
        ## 添加基本的数据增强，对图片和标签保持一致
        ## 添加固定尺度的随机裁剪，使用最近邻缩放（不产生新的灰度值）+裁剪
        image = cv2.resize(image,(self.imagesize,self.imagesize),interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label,(self.imagesize,self.imagesize),interpolation=cv2.INTER_NEAREST)
        offsetx = np.random.randint(self.imagesize-self.cropsize)
        offsety = np.random.randint(self.imagesize-self.cropsize)
        image = image[offsety:offsety+self.cropsize,offsetx:offsetx+self.cropsize]
        label = label[offsety:offsety+self.cropsize,offsetx:offsetx+self.cropsize]
       
        return self.transform(image),label ##只对image做预处理操作（转成tensor，并进行归一化操作），获得可以输入网络的图片

    def __len__(self): ##统计数据集大小
        return len(self.samples)

if __name__ == '__main__':
    filetxt = "dataset/train.txt"
    imagesize = 256
    cropsize = 224
    mydataset = SegDataset(filetxt,imagesize,cropsize)
    
    # print(mydataset.samples)
    # print(mydataset.__len__())
    
    image,label = mydataset.__getitem__(0)

    srcimage = cv2.imread(open(filetxt,'r').readlines()[0].strip().split(' ')[0])
    cv2.namedWindow("image",0)
    cv2.imshow("image",srcimage)
    cv2.namedWindow("cropimage",0)
    cv2.imshow("cropimage",((image.numpy()*0.5+0.5)*255).astype(np.uint8).transpose(1,2,0))
    cv2.waitKey(0)

    print(image.shape)
    print(label.shape)
