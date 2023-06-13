import torchvision
import torch
from PIL import Image
import torch.nn as nn
from Alexnet import AlexNet
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as data
import os

def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

model = AlexNet(43)
model.load_state_dict(torch.load('./Model/pytorch_classification_resnetTS.pth'))
img_transforms = transforms.Compose([
       transforms.Resize([227, 227]),
       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
       transforms.ToTensor()
   ])

#data_loader = data.DataLoader(data_test,batch_size=256)
#print(len(data_loader))
# 生成类标签
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }




numClasses = 43
num = range(numClasses)
labels = []
for i in num:
    labels.append(str(i))
labels = sorted(labels)
for i in num:
    labels[i] = int(labels[i])
model.eval()

with torch.no_grad():
    imglist = getFileList("./test", [], 'jpg')
    for path in imglist:
        img = Image.open(path)
        img = img_transforms(img)
        img = img.resize(1, 3,227, 227)  # 转换图片为这种格式，为了让图片能够作为模型的输入
        pred_result = model(img)
        pred_softmax = torch.softmax(pred_result, dim=1)
        pred_probality, pred_tags = torch.max(pred_softmax, dim=1)
        pred_probality = pred_probality.cpu().numpy()
        pred_tags = pred_tags.cpu().numpy()
        pred_probality = pred_probality[0]
        pred_label = pred_tags[0]
        pred_label = labels[pred_label]
        pre_class = classes[pred_label]
        # 输出结果：类标签序号：识别结果[准确率]
        print(str(pred_label) + ":" + pre_class + "[" + str(pred_probality) +"]")
