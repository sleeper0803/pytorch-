from matplotlib import image
import torch
from torch.autograd import Variable
from torchvision import datasets,transforms,utils,models
import os
import matplotlib.pyplot as plt
import time 
from torch.hub import load_state_dict_from_url


#matplotlib inline 

print(torch.cuda.is_available())
Use_gpu=torch.cuda.is_available()
#Use_gpu=False

data_dir="DogsVSCats"
data_transform={x:transforms.Compose([transforms.Scale([224,224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
                    for x in ["train","valid"]}
 
image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                transform=data_transform[x]) for x in ["train","valid"]}

dataloader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],batch_size=16,
                shuffle=True,) for x in["train","valid"]}

X_example,Y_example=next(iter(dataloader["train"]))
Index_classes=image_datasets["train"].class_to_idx
example_classes=image_datasets["train"].classes

#print(u"X_example 个数{}".format(len(X_example)))
#print(u"Y_example 个数{}".format(len(Y_example)))
#print(Index_classes)
#print(example_classes)

model=models.vgg16(pretrained=True)
for parma in model.parameters():
    parma.requires_grad=False


model.classifier=torch.nn.Sequential(
                 torch.nn.Linear(in_features=7*7*512,out_features=4096),
                 torch.nn.ReLU6(inplace=True),
                 torch.nn.Dropout(p=0.5),
                 torch.nn.Linear(in_features=4096, out_features=4096),
                 torch.nn.ReLU(inplace=True),
                 torch.nn.Dropout(p=0.5),
                 torch.nn.Linear(in_features=4096, out_features=2)
)
if Use_gpu:
    #model=model.cuda()
    model=model.to('cuda')
    #model=model.gpu()

#图片可视化
#img=utils.make_grid(X_example)
#img=img.numpy().transpose([1,2,0])
#print([example_classes[i] for i in Y_example])
#plt.imshow(img)
#plt.show()

cost=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.classifier.parameters())

loss_f=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.classifier.parameters(),lr=0.00001)


epoch_n=10
time_start=time.time()
for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch,epoch_n-1))
    print("*"*50)

    for phase in ["train","valid"]:
        if phase=="train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)
        running_loss=0.0
        running_corrects=0.0

        for batch,data in enumerate(dataloader[phase],1):
            X,Y=data
            if Use_gpu:
                X,Y=Variable(X.cuda()),Variable(Y.cuda())
            else:
                X,Y=Variable(X),Variable(Y)

            Y_pred=model(X)
            _,pred=torch.max(Y_pred.data,1)
            optimizer.zero_grad()
            loss=loss_f(Y_pred,Y)

            if phase=="train":
                loss.backward()
                optimizer.step()
            
            running_loss+=loss.data
            running_corrects+=torch.sum(pred==Y.data)

            if batch%500==0 and phase=="train":
                print("Batch {},Train Loss:{:.4f},Train ACC:{:.4f}".format(
                    batch,running_loss/batch,100*running_corrects/(16*batch)))
        epoch_loss=running_loss*16/len(image_datasets[phase])
        epoch_acc=100*running_corrects/len(image_datasets[phase])

        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss,epoch_acc))
    time_Dur=time.time()-time_start
    print(time_Dur)