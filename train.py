#coding:utf8

# or create issues
# =============================================================================
from __future__ import print_function, division

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import os
from net import simpleNet5
# from net import VGGNet, FCNs
from dataset import SegDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

writer = SummaryWriter('log') #可视化
batchsize = 64
epochs = 200
imagesize = 256 #缩放图片大小
cropsize = 224 #训练图片大小
train_data_path = 'dataset/train.txt'  #训练数据集
val_data_path = 'dataset/val.txt'  #验证数据集

# 将label从[0,255]变成[0,1]
# imagepaths=os.listdir('dataset/label/')
# for imagepath in imagepaths:
#     img=cv2.imread(os.path.join('dataset/label/',imagepath))
#     mask=(img/255).astype(np.uint8)
#     cv2.imwrite(os.path.join('dataset/label/',imagepath),mask)
# 数据预处理
data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

# 图像分割数据集
train_dataset = SegDataset(train_data_path,imagesize,cropsize,data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_dataset = SegDataset(val_data_path,imagesize,cropsize,data_transform)
val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=True)

image_datasets = {}
image_datasets['train'] = train_dataset
image_datasets['val'] = val_dataset
dataloaders = {}
dataloaders['train'] = train_dataloader
dataloaders['val'] = val_dataloader
for data in dataloaders['train']:
    print(data)

# 定义网络，优化目标，优化方法
device = torch.device('cpu')
net = simpleNet5().to(device)
# vgg_model=VGGNet(requires_grad=True,show_params=True)
# fcn_model=FCNs(pretrained_net=vgg_model,n_class=2)
# net=fcn_model.to(device)
# torch.save(net, 'model.pt')
criterion = nn.CrossEntropyLoss() #使用softmax loss损失，输入label是图片
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) #每50个epoch，学习率衰减


if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

for epoch in range(1, epochs+1):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            net.train(True)  # Set model to training mode
        else:
            net.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_accs = 0.0

        n = 0
        for data in dataloaders[phase]:
            imgs, labels = data
            img, label = imgs.to(device).float(), labels.to(device).float()
            output = net(img)
            loss = criterion(output, label.long()) #得到损失

            output_mask = output.cpu().data.numpy().copy()
            output_mask = np.argmax(output_mask, axis=1)
            y_mask = label.cpu().data.numpy().copy()
            acc = (output_mask == y_mask) #计算精度
            acc = acc.mean()

            optimizer.zero_grad()
            if phase == 'train':
            # 梯度置0，反向传播，参数更新
                loss.backward()
                optimizer.step()

            running_loss += loss.data.item()
            running_accs += acc
            n += 1

        epoch_loss = running_loss / n
        epoch_acc = running_accs / n

        if phase == 'train':
            writer.add_scalar('data/trainloss', epoch_loss, epoch)
            writer.add_scalar('data/trainacc', epoch_acc, epoch)
            print('train epoch_{} loss='+str(epoch_loss).format(epoch))
            print('train epoch_{} acc='+str(epoch_acc).format(epoch))
        else:
            writer.add_scalar('data/valloss', epoch_loss, epoch)
            writer.add_scalar('data/valacc', epoch_acc, epoch)
            print('val epoch_{} loss='+str(epoch_loss).format(epoch))
            print('val epoch_{} acc='+str(epoch_acc).format(epoch))


    if epoch % 10 == 0:

        torch.save(net, 'checkpoints/model_epoch_{}.pth'.format(epoch))
        print('checkpoints/model_epoch_{}.pth saved!'.format(epoch))

torch.save(net,'model.pt')
writer.export_scalars_to_json("./all_scalars.json")
writer.close()