#coding:utf8
# =============================================================================
import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F
import numpy as np

data_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

modelpath = 'checkpoints/model_epoch_40.pth' #模型目录
net = torch.load(modelpath,map_location='cpu')
net.eval() #设置为推理模式，不会更新模型的k，b参数

cap = cv2.VideoCapture(0)
torch.no_grad() #停止autograd模块的工作，加速和节省显存
##读摄像头信息
while True:
    ok, img = cap.read()
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    imgblob = data_transforms(image).unsqueeze(0) #填充维度，从3维到4维
    predict = F.softmax(net(imgblob)).cpu().data.numpy().copy() #获得原始网络输出，多通道
    predict = np.argmax(predict, axis=1) #得到单通道label
    result = np.squeeze(predict) #降低维度，从4维到3维
    print(np.max(result)) 
    result = (result*255).astype(np.uint8) #灰度拉伸，方便可视化

    resultimage = image.copy()
    for y in range(0,result.shape[0]): 
        for x in range(0,result.shape[1]):
            if result[y][x] == 0:
                resultimage[y][x] = (255,255,255)

    combineresult = np.concatenate([image,resultimage],axis=1)
    cv2.imshow('video', combineresult)
    # 点击窗口关闭按钮退出程序
    if cv2.getWindowProperty('video', cv2.WND_PROP_AUTOSIZE) < 1:
        break
cap.release()
cv2.destroyAllWindows()





