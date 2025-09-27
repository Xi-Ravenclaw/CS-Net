import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

#Title: "CS-Net: Convolutional Spider Neural Network for Surface-EMG-Based hybrid Gesture Recognition"

#Authors:  "Xi ZHANG, Jiannan CHEN, Lei LIU, Fuchun SUN"

#DOI: 10.1088/1741-2552/ae0c38

#Github URL: https://github.com/Xi-Ravenclaw/CS-Net

#License：Apache-2.0

class SpiderBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=(3,3)):
        super(SpiderBlock, self).__init__()
        
        self.conv11= nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=64,           kernel_size=kernel_size,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU()) 
        self.conv12= nn.Sequential(nn.Conv2d(in_channels=64,          out_channels=out_channels, kernel_size=kernel_size,stride=1,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU())
        self.dropout1=nn.Dropout2d(0.2)
        
        self.conv21= nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=64,           kernel_size=kernel_size,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.conv22= nn.Sequential(nn.Conv2d(in_channels=64,          out_channels=out_channels, kernel_size=kernel_size,stride=1,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU())
        self.dropout2=nn.Dropout2d(0.2)
  
    def forward(self, input1,input2):
        input=torch.cat([input1,input2],dim=3)

        x = self.conv11(input)
        x = self.conv12(x)
        x = self.dropout1(x)
        
        y = self.conv21(input)
        y = self.conv22(y)
        y = self.dropout2(y)
        return (x,y)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=(3,3)):
        super(TransitionBlock, self).__init__()
        
        self.conv1= nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=1),nn.ReLU())
        self.pool1=nn.MaxPool2d(kernel_size=(1,2), stride=(1,2),padding=(0,0)) 
        
        self.conv2= nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,padding=1),nn.ReLU())
        self.pool2=nn.MaxPool2d(kernel_size=(1,2), stride=(1,2),padding=(0,0))  
        
    def forward(self,input1,input2):
        x = self.conv1(input1)
        x = self.pool1(x)
        
        y = self.conv2(input2)
        y = self.pool2(y)
        return x,y

class Net(nn.Module):
    def __init__(self,class_number,flatten_shape_number):
        super(Net,self).__init__()

        self.filter1= nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=1,padding=1),nn.ReLU())
        self.filter2= nn.Sequential(nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=1,padding=1),nn.ReLU())
        

        self.spider11=SpiderBlock(in_channels=64, out_channels=64, )
        self.spider21=SpiderBlock(in_channels=64, out_channels=64, )
        
        self.Tran11=TransitionBlock(in_channels=64, out_channels=32, )
        self.Tran21=TransitionBlock(in_channels=64, out_channels=32, )
        
        self.spider12=SpiderBlock(in_channels=32, out_channels=32, )
        self.spider22=SpiderBlock(in_channels=32, out_channels=32, )
        
        self.Tran12=TransitionBlock(in_channels=32, out_channels=16, ) 
        self.Tran22=TransitionBlock(in_channels=32, out_channels=16, ) 


        self.flatten_shape = flatten_shape_number #you can find this value from python error message
        self.flatten=nn.Sequential(nn.Flatten(1,-1))
        
        self.fcl1= nn.Sequential(nn.Linear(self.flatten_shape, 512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2))
        self.fcl3= nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2))
        self.fcl4= nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2))
        self.output= nn.Sequential(nn.Linear(128,class_number),nn.Softmax(dim=1))
           
    def forward(self,raw_data,fft_data):
        x_input=self.filter1(raw_data)
        y_input=self.filter2(fft_data)
        
        x0,x1=self.spider11(x_input,y_input)
        y0,y1=self.spider21(x_input,y_input)
        x0,x1=self.Tran11(x0,x1)
        y0,y1=self.Tran21(y0,y1)
        
        x0,x1=self.spider12(x0,x1)
        y0,y1=self.spider22(y0,y1)
        x0,x1=self.Tran12(x0,x1)
        y0,y1=self.Tran22(y0,y1)
        
        output2FC=torch.cat([x0,x1,y0,y1],dim=2)
        
        x=self.flatten(output2FC)
        x=self.fcl1(x)
        x=self.fcl3(x)
        x=self.fcl4(x)
        x=self.output(x)
        return x