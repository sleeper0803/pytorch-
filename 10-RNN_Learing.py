import torch
from torch.autograd.variable import VariableMeta
from torch.utils.data import dataset
import torchvision
from torchvision import transforms,datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt




Transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5],std=[0.5])])
DataSetTrain=datasets.MNIST(root="./dataMNIST/",
                            transform=Transform,train=True,download=True)
DataSetTest=datasets.MNIST(root="./dataMNIST/",
                            transform=Transform,train=False)

TrainLoad=torch.utils.data.DataLoader(dataset=DataSetTrain,batch_size=64,shuffle=True)
TestLoad=torch.utils.data.DataLoader(dataset=DataSetTest,batch_size=64,shuffle=True)

Images,Label=next(iter(TrainLoad))
ImagesExample=torchvision.utils.make_grid(Images)
ImagesExample=ImagesExample.numpy().transpose(1,2,0)
#plt.imshow(ImagesExample)
#plt.show()
Mean=[0.5,0.5,0.5]
Std=[0.5,0.5,0.5]
ImagesExample=ImagesExample*Std+Mean
plt.imshow(ImagesExample)
plt.show()

class myRNN(torch.nn.Module):
    def __init__(self):
        super(myRNN,self).__init__()
        # 定义RNN网络,输入单个数字.隐藏层size为[feature, hidden_size]
        self.rnn=torch.nn.RNN(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True # 注意这里用了batch_first=True 所以输入形状为[batch_size, time_step, feature]
        )
        # 定义一个全连接层,本质上是令RNN网络得以输出
        self.output=torch.nn.Linear(128,10)
    # 定义前向传播函数
    def forward(self,input):
        # 给定一个序列input,每个input.size=[batch_size, feature].
        # 同时给定一个h_state初始状态,RNN网络输出结果并同时给出隐藏层输出
        output,_=self.rnn(input,None)
        output=self.output(output[:,-1,:])
        return output

Model=myRNN()

Optimizer=torch.optim.Adam(Model.parameters())
Loss_f=torch.nn.CrossEntropyLoss()

EpochN=10

for Epoch in range(EpochN):
    RunningLoss=0.0
    RunningCorrect=0.0
    TestingLoss=0.0
    TestingCorrect=0.0
    print("Epoch {}/{}".format(Epoch,EpochN))
    print("*"*50)

    for data in TrainLoad:
        Xtrain,Ytrain=data
        Xtrain=Xtrain.view(-1,28,28)
        Xtrain,Ytrain=Variable(Xtrain),Variable(Ytrain)
        Ypred=Model(Xtrain)
        Loss=Loss_f(Ypred,Ytrain)
        _,pred=torch.max(Ypred.data,1)

        Optimizer.zero_grad()
        Loss.backward()
        Optimizer.step()
        RunningLoss+=Loss.data
        RunningCorrect+=torch.sum(pred==Ytrain.data)

    for data in TestLoad:
        Xtest,Ytest=data
        Xtest=Xtest.view(-1,28,28)
        Xtest,Ytest=Variable(Xtest),Variable(Ytest)
        output=Model(Xtest)
        _,pred=torch.max(output.data,1)
        TestingCorrect+=torch.sum(pred==Ytest)

    print("Loss is :{:.4f},Train Accuracy is : {:.4f}%,Test Accuracy is :{:.4f}"
                .format(RunningLoss/len(TrainLoad),
                100*RunningCorrect/len(DataSetTrain),
                100*TestingCorrect/len(DataSetTest)))

