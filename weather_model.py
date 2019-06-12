import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

#h-sigmoid
class Hsigmoid(nn.Module):
    def forward(self, x):
        out=F.relu6(x+3,inplace=True)/6
        return out

#Squeeze and Excitation
class SE(nn.Module):
    def __init__(self,in_size,reduction=4):
        super(SE, self).__init__()
        self.se=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size,in_size//reduction,1,1,bias=False),
            nn.BatchNorm2d(in_size//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size//reduction,in_size,1,1,bias=False),
            nn.BatchNorm2d(in_size),
            Hsigmoid()
        )
    def forward(self, x):
        return x*self.se(x)

#h-swish
class Hswish(nn.Module):
    def forward(self, x):
        out=x*F.relu6(x+3,inplace=True)/6
        return out

#building blocks
class Blocks(nn.Module):
    def __init__(self,in_size,expand_size,out_size,stride):
        super(Blocks,self).__init__()
        self.stride=stride
        self.se=SE(out_size)
        self.conv1=nn.Conv2d(in_size,expand_size,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(expand_size)
        self.hs1=Hswish()
        self.conv2=nn.Conv2d(expand_size,expand_size,3,stride,1,groups=expand_size,bias=False)
        self.bn2=nn.BatchNorm2d(expand_size)
        self.hs2=Hswish()
        self.conv3=nn.Conv2d(expand_size,expand_size,3,1,1,groups=expand_size,bias=False)
        self.bn3=nn.BatchNorm2d(expand_size)
        self.hs3=Hswish()
        self.conv4=nn.Conv2d(expand_size,out_size,1,1,bias=False)
        self.bn4=nn.BatchNorm2d(out_size)
        self.skipcon=nn.Sequential(
            nn.Conv2d(in_size,out_size,1,1,bias=False),
            nn.BatchNorm2d(out_size)
        )
    def forward(self, x):
        out=self.hs1(self.bn1(self.conv1(x)))
        out=self.hs2(self.bn2(self.conv2(out)))
        out=self.hs3(self.bn3(self.conv3(out)))
        out=self.bn4(self.conv4(out))
        out=self.se(out)
        out=out+self.skipcon(x) if self.stride==1 else out
        return out

#define the model
class Model(nn.Module):
    def __init__(self,scale=1):
        super(Model,self).__init__()
        self.conv1=nn.Conv2d(3,int(16*scale),3,2,1,bias=False)#out=112
        self.bn1=nn.BatchNorm2d(int(16*scale))
        self.hs1=Hswish()
        self.block=nn.Sequential(
            Blocks(int(16*scale),int(32*scale),int(24*scale),stride=2), #out=56
            Blocks(int(24*scale),int(48*scale),int(36*scale),stride=1),#out=56
            Blocks(int(36*scale),int(72*scale),int(48*scale),stride=2),#out=28
            Blocks(int(48*scale),int(96*scale),int(64*scale),stride=1),#out=28
            Blocks(int(64*scale),int(128*scale),int(96*scale),stride=2),#out=14
            Blocks(int(96*scale),int(192*scale),int(192*scale),stride=1),#out=14
        )
        self.conv2=nn.Conv2d(int(192*scale),int(728*scale),1,1,bias=False)
        self.bn2=nn.BatchNorm2d(int(728*scale))
        self.hs2=Hswish()
        self.fc=nn.Linear(int(728*scale),6)
        self.out=nn.LogSoftmax(dim=1)
        self.init_params()
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        out=self.hs1(self.bn1(self.conv1(x)))
        out=self.block(out)
        out=self.hs2(self.bn2(self.conv2(out)))
        out=F.avg_pool2d(out,14)#
        out=out.view(out.size(0),-1)
        out=self.out(self.fc(out))
        return out

# net=Model(1)

# from torchsummary import summary
# summary(net.to('cuda'),(3,224,224))