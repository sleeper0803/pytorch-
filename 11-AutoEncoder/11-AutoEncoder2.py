'''
    @第十一章，通过卷积变换实现自动编码功能
    @时间：2021-11-26
    @此小节中使用了cuda计算
'''
import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

Transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5],std=[0.5])])
DataSetTrain=datasets.MNIST(root="./dataMNIST/",
                            transform=Transform,train=True,download=True)
DataSetTest=datasets.MNIST(root="./dataMNIST/",
                            transform=Transform,train=False)

TrainLoad=torch.utils.data.DataLoader(dataset=DataSetTrain,batch_size=4,shuffle=True)
TestLoad=torch.utils.data.DataLoader(dataset=DataSetTest,batch_size=4,shuffle=True)

Images,Label=next(iter(TrainLoad))
print(Images.shape)
ImagesExample=torchvision.utils.make_grid(Images)
ImagesExample=ImagesExample.numpy().transpose(1,2,0)
#plt.imshow(ImagesExample)
#plt.show()
#Mean=[0.5,0.5,0.5]
#Std=[0.5,0.5,0.5]
Mean=[0.5]
Std=[0.5]
ImagesExample=ImagesExample*Std+Mean
plt.imshow(ImagesExample)
plt.show()
NoiseImage=ImagesExample+0.5*np.random.randn(*ImagesExample.shape)
NoiseImage=np.clip(NoiseImage,0.,1.)
plt.imshow(NoiseImage)
plt.show()

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.decoder=torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1),
        )
    def forward(self,input):
        output=self.encoder(input)
        output=self.decoder(output)
        return output

Model=AutoEncoder()
Use_gpu=torch.cuda.is_available()
if Use_gpu:
    Model=Model.cuda()
print(Model)

Optimizer=torch.optim.Adam(Model.parameters())
Loss_f=torch.nn.MSELoss()

EpochN=10

for Epoch in range(EpochN):
    RunningLoss=0.0

    print("Epoch {}/{}".format(Epoch,EpochN))
    print("*"*50)

    for data in TrainLoad:
        Xtrain,_=data

        NoiseXtrain=Xtrain+0.5*torch.randn(Xtrain.shape)
        NoiseXtrain=torch.clamp(NoiseXtrain,0.,1.)

        Xtrain,NoiseXtrain=Variable(Xtrain.cuda()),Variable(NoiseXtrain.cuda())
        TrainPred=Model(NoiseXtrain)
        Loss=Loss_f(TrainPred,Xtrain)

        Optimizer.zero_grad()
        Loss.backward()
        Optimizer.step()
        RunningLoss+=Loss.data


    for data in TestLoad:
        Xtest,_=data
        TestXTest=Xtest+0.5*torch.randn(Xtest.shape)
        TestXTest=torch.clamp(TestXTest,0.,1.)
        TestXTest=Variable(TestXTest.cuda())
        output=Model(TestXTest)
        #TestPred=torch.max(output.data,1)

        TestPred2=output.data.view(-1,1,28,28)
        TestImgShow=torchvision.utils.make_grid(TestPred2)
        TestImgShow=TestImgShow.cpu()
        TestImgShow=TestImgShow.numpy().transpose(1,2,0)
        TestImgShow=TestImgShow*Std+Mean
        TestImgShow=np.clip(TestImgShow,0.,1.)

        plt.imshow(TestImgShow)
        plt.show()

    print("{} Loss:{:.4f} Acc:{:.4f}%".format(RunningLoss/len(DataSetTrain)))
