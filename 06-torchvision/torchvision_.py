import torch
#from torch.utils.data import dataset
from torchvision import datasets , transforms,utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab


'''
transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5],
                        std=[0.5])])
'''
transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.5],[0.5])])


data_train=datasets.MNIST(root="./dataMNIST/", 
                    transform=transform,
                    train=True,
                    download=True)

data_test=datasets.MNIST(root="./dataMNIST/",
                    transform=transform,
                    train=False)

data_loader_train=torch.utils.data.DataLoader(dataset=data_train,
batch_size=64,shuffle=True)

data_loader_test=torch.utils.data.DataLoader(dataset=data_test,
batch_size=64,shuffle=True)

images,labels=next(iter(data_loader_train))
img=utils.make_grid(images)
img=img.numpy().transpose(1,2,0)

std=[0.5]
mean=[0.5]
img=img*std+mean
print([labels[i] for i in range(64)])
plt.imshow(img)
pylab.show()
a=10