'''
    @第十一章，线性编码代码实现
    @时间：2021-11-26
'''
import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
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
Mean=[0.5,0.5,0.5]
Std=[0.5,0.5,0.5]
#Mean=[0.5]
#Std=[0.5]
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
            torch.nn.Linear(28*28,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
        )
        self.decoder=torch.nn.Sequential(
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,28*28),
            torch.nn.ReLU(),
        )
    def forward(self,input):
        output=self.encoder(input)
        output=self.decoder(output)
        return output

Model=AutoEncoder()
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

        Xtrain,NoiseXtrain=Variable(Xtrain.view(-1,28*28)),Variable(NoiseXtrain.view(-1,28*28))
        TrainPred=Model(NoiseXtrain)
        Loss=Loss_f(TrainPred,Xtrain)

        Optimizer.zero_grad()
        Loss.backward()
        Optimizer.step()
        RunningLoss+=Loss.data
    

    for data in TestLoad:
        Xtest,Ytest=data

        TestXtrain=Xtest+0.5*torch.randn(Xtest.shape)
        '''
        TestXBeforeNoise=torchvision.utils.make_grid(Xtest)
        TestXBeforeNoise=TestXBeforeNoise.numpy().transpose(1,2,0)
        TestXBeforeImage=torchvision.utils.make_grid(TestXtrain)
        TestXBeforeImage=TestXBeforeImage.numpy().transpose(1,2,0)
        plt.imshow(TestXBeforeNoise)
        plt.show()
        plt.imshow(TestXBeforeImage)
        plt.show()
        '''
        TestXtrain=torch.clamp(TestXtrain,0.,1.)
        TestXtrain=TestXtrain.view(-1,28*28)
        TestPred=Model(TestXtrain)

        TestPred2=TestPred.data.view(-1,1,28,28)
        TestImgShow=torchvision.utils.make_grid(TestPred2)
        TestImgShow=TestImgShow.numpy().transpose(1,2,0)
        TestImgShow=TestImgShow*Std+Mean
        TestImgShow=np.clip(TestImgShow,0.,1.)

        plt.imshow(TestImgShow)
        plt.show()

    print("Loss:{:.4f}".format(RunningLoss/len(DataSetTrain)))
