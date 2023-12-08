import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch.quantization import fuse_modules
from torch.nn.quantized import FloatFunctional
import torch.nn.quantized as nnq
import torch.nn.functional as F
from torch import Tensor
from torch.quantization import fuse_modules

def conv2d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

class BottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(BottleNeck, self).__init__()
        self.layer1 = conv2d(in_channels, out_channels, kernel_size)
        self.layer2 = conv2d(out_channels, out_channels, kernel_size)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer1(x)
        y = self.layer2(y)
        x = self.conv1x1(x)
        return x + y
    

class BasicBlockSem(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicBlockSem, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_planes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        out = self.ca(out) * out
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    

class SASceneNet(nn.Module):
    def __init__(self, arch, completePath, scene_classes:int=6, semantic_classes=151, bottle_neck = BottleNeck, quant=False):
        super(SASceneNet, self).__init__()
        
        self.quant = quant  
        self.scene_classes = scene_classes

        if arch == 'ResNet-18':
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            size_fc_RGB = 512
            sizes_lastConv = [512, 512, 512]   
        elif arch == 'ResNet-50':
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            size_fc_RGB = 2048
            sizes_lastConv = [2048, 1024, 1024]

        
        self.res18branch = self.ResNet18(scene_classes, bottle_neck)

        self.in_block_sem = nn.Sequential(
            nn.Conv2d(semantic_classes+1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_block_sem_1 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_2 = BasicBlockSem(128, 256, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_3 = BasicBlockSem(256, 512, kernel_size=3, stride=2, padding=1)


        self.fc_SEM = nn.Linear(512, scene_classes) 


        self.lastConvRGB1 = nn.Sequential(
            nn.Conv2d(sizes_lastConv[0], sizes_lastConv[1], kernel_size=3, bias=False),
            nn.BatchNorm2d(sizes_lastConv[1]),
            nn.ReLU(inplace=True),
        )
        self.lastConvRGB2 = nn.Sequential(
            nn.Conv2d(sizes_lastConv[2], 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.lastConvSEM1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.lastConvSEM2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.avgpool7 = nn.AvgPool2d(7, stride=1)
        self.avgpool3 = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(1024, scene_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.load_weights(completePath) 

        
    def load_weights(self, completePath):
        checkpoint = torch.load(completePath)
        self.load_state_dict(checkpoint['state_dict'], strict=False)


    def forward_sascene_sem(self, e4, sem):

        y = self.in_block_sem(sem)
        y1 = self.in_block_sem_1(y)
        y2 = self.in_block_sem_2(y1)
        y3 = self.in_block_sem_3(y2)

        e5 = self.lastConvRGB1(e4)
        e6 = self.lastConvRGB2(e5)

        y4 = self.lastConvSEM1(y3)
        y5 = self.lastConvSEM2(y4)

        e7 = e6 * self.sigmoid(y5)
        return e7


    def forward_sascene_classifier(self, e7):

        e8 = self.avgpool3(e7)
        act = e8.view(e8.size(0), -1)
        act = self.dropout(act)
        act = self.fc(act)

        return act
    
    def forward_sascene(self, xnsem):
        x = xnsem[0]
        sem = xnsem[1]
        e4 = self.res18branch.forward_sascene_res18(x)
        e7 = self.forward_sascene_sem(e4, sem)
        act = self.forward_sascene_classifier(e7)

        return act, e7

    def forward(self, x):
        self.forward_sascene(x)

    def loss(self, x, target):

        assert (x.shape[0] == target.shape[0])

        loss = self.criterion(x, target.long())

        return loss
    
    class ResNet18(nn.Module):
        def __init__(self, num_classes, bottle_neck: nn.Module = BottleNeck):
            super(SASceneNet.ResNet18, self).__init__()

            self.conv1 = conv2d(3, 64, 7, 2, 3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(3, 2)

            self.layer1 = nn.Sequential(bottle_neck(64, 64), bottle_neck(64, 64), nn.MaxPool2d(2, 2))
            self.layer2 = nn.Sequential(bottle_neck(64, 128), bottle_neck(128, 128), nn.MaxPool2d(2, 2))
            self.layer3 = nn.Sequential(bottle_neck(128, 256), bottle_neck(256, 256), nn.MaxPool2d(2, 2))
            self.layer4 = nn.Sequential(bottle_neck(256, 512), bottle_neck(512, 512), nn.MaxPool2d(2, 2))

            self.avgpool = self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def forward_sascene_res18(self, x: Tensor) -> Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            e4 = self.layer4(x)

            x = self.avgpool(e4)
            x = torch.flatten(x, 1)

            return e4

        def forward(self, x):
            return self.forward_sascene_res18(x)


class QuantizableBottleNeck(BottleNeck):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(QuantizableBottleNeck, self).__init__(in_channels, out_channels, kernel_size)
        self.float_functional = FloatFunctional()

    def fuse_model(self) -> None:
        fuse_modules(self.layer1, ["0", "1", "2"], inplace=True)
        fuse_modules(self.layer2, ["0", "1", "2"], inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer1(x)
        y = self.layer2(y)
        x = self.conv1x1(x)
        return self.float_functional.add(x, y)

class QuantizableResnet18(SASceneNet):

    def __init__(self, arch, completePath, scene_classes:int=6, semantic_classes=151, bottle_neck=QuantizableBottleNeck, quant=False):
        super(QuantizableResnet18, self).__init__(arch=arch, completePath=completePath, semantic_classes=149, bottle_neck=QuantizableBottleNeck)
        self.quant1 = torch.quantization.QuantStub()
        self.dequant1 = torch.quantization.DeQuantStub()

        self.quant2 = torch.quantization.QuantStub()
        self.dequant2 = torch.quantization.DeQuantStub()

        self.quant3 = torch.quantization.QuantStub()
        self.dequant3 = torch.quantization.DeQuantStub()

        self.quant4 = torch.quantization.QuantStub()
        self.dequant4 = torch.quantization.DeQuantStub()
        
        self.quant5 = torch.quantization.QuantStub()
        self.dequant5 = torch.quantization.DeQuantStub()
        self.dequant6 = torch.quantization.DeQuantStub()
        self.dequant7 = torch.quantization.DeQuantStub()
        
    
    def forward(self, xnsem):
        x, sem = xnsem[0], xnsem[1]

        x = self.quant1(x)
        e4 = self.res18branch.forward_sascene_res18(x)
        x = self.dequant1(x)
        
        sem = self.quant2(sem)
        e7 = self.forward_sascene_sem(e4, sem)

        sem = self.dequant2(sem)
        e4 = self.dequant4(e4)

        act = self.forward_sascene_classifier(e7)
        act = self.dequant6(act)
        e7 = self.dequant3(e7)

        return act    
    
    def forward_sascene(self, xnsem):
        x = xnsem[0]
        sem = xnsem[1]
       
        x = self.quant1(x)
        e4 = self.res18branch.forward_sascene_res18(x)
        x = self.dequant1(x)

        sem = self.quant2(sem)
        e7 = self.forward_sascene_sem(e4, sem)
        sem = self.dequant2(sem)
        e4 = self.dequant4(e4)

        act = self.forward_sascene_classifier(e7)
        act = self.dequant6(act)
        e7 = self.dequant3(e7)
        
        return act, e7
    
    
    def forward_sascene_sem(self, e4, sem):

        y = self.in_block_sem(sem)
        y1 = self.in_block_sem_1(y)
        y2 = self.in_block_sem_2(y1)
        y3 = self.in_block_sem_3(y2)

        e5 = self.lastConvRGB1(e4)
        e6 = self.lastConvRGB2(e5)

        y4 = self.lastConvSEM1(y3)
        y5 = self.lastConvSEM2(y4)
        e7 = e6 * self.sigmoid(y5)

        return e7


    def fuse_model(self) -> None:
        fuse_modules(self.res18branch.conv1, ['0', '1', '2'], inplace=True)
        for m in self.modules():
            if type(m) is QuantizableBottleNeck:
                m.fuse_model()
