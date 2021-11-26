from matplotlib import image
import torch
from torch.autograd import Variable
from torchvision import datasets,transforms,utils,models
import os
import matplotlib.pyplot as plt
import time 
#from torch.hub import load_state_dict_from_url
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


model_1=models.vgg16(pretrained=True)
model_2=models.resnet50(pretrained=True)

for parma in model_1.parameters():
    parma.requires_grad=False
model_1.classifier=torch.nn.Sequential(
                 torch.nn.Linear(in_features=7*7*512,out_features=4096),
                 torch.nn.ReLU6(inplace=True),
                 torch.nn.Dropout(p=0.5),
                 torch.nn.Linear(in_features=4096, out_features=4096),
                 torch.nn.ReLU(inplace=True),
                 torch.nn.Dropout(p=0.5),
                 torch.nn.Linear(in_features=4096, out_features=2)
)

for parma in model_2.parameters():
    parma.requires_grad=False
model_2.fc=torch.nn.Linear(2048,2)

if Use_gpu:
    model_1=model_1.to('cuda')
    model_2=model_2.to('cuda')

loss_f_1=torch.nn.CrossEntropyLoss()
optimizer_1=torch.optim.Adam(model_1.classifier.parameters(),lr=0.00001)
loss_f_2=torch.nn.CrossEntropyLoss()
optimizer_2=torch.optim.Adam(model_2.fc.parameters(),lr=0.00001)
weight_1=0.6
weight_2=0.4

epoch_n=10
time_start=time.time()
for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch,epoch_n-1))
    print("*"*50)

    for phase in ["train","valid"]:
        if phase=="train":
            print("Training...")
            model_1.train(True)
            model_2.train(True)
        else:
            print("Validing...")
            model_1.train(False)
            model_2.train(False)
        running_loss_1=0.0
        running_corrects_1=0.0
        running_loss_2=0.0
        running_corrects_2=0.0
        Blending_running_corrects=0.0

        for batch,data in enumerate(dataloader[phase],1):
            X,Y=data
            if Use_gpu:
                X,Y=Variable(X.cuda()),Variable(Y.cuda())
            else:
                X,Y=Variable(X),Variable(Y)

            Y_pred_1=model_1(X)
            Y_pred_2=model_2(X)
            Blending_Y_pred=Y_pred_1*weight_1+Y_pred_2*weight_2

            _,pred_1=torch.max(Y_pred_1.data,1)
            _,pred_2=torch.max(Y_pred_2.data,1)
            _,Blending_pred=torch.max(Blending_Y_pred.data,1)

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss_1=loss_f_1(Y_pred_1,Y)
            loss_2=loss_f_2(Y_pred_2,Y)

            if phase=="train":
                loss_1.backward()
                loss_2.backward()
                optimizer_1.step()
                optimizer_2.step()
            
            running_loss_1 +=loss_1.data
            running_loss_2 +=loss_2.data
            running_corrects_1+=torch.sum(pred_1==Y.data)
            running_corrects_2+=torch.sum(pred_2==Y.data)
            Blending_running_corrects+=torch.sum(Blending_pred==Y.data)

            if batch%500==0 and phase=="train":
                print("Batch {}, Modell Train Loss:{:.4f},Train ACC:{:.4f}\
                                 Model2 Train Loss:{:.4f},Train ACC:{:.4f}\
                                 Blending Train ACC:{:.4f}".format(
                    batch,running_loss_1/batch,100*running_corrects_1/(16*batch),
                          running_loss_2/batch,100*running_corrects_2/(16*batch),
                          100*Blending_running_corrects/(16*batch),
                    ))
        epoch_loss_1=running_loss_1*16/len(image_datasets[phase])
        epoch_loss_2=running_loss_2*16/len(image_datasets[phase])
        epoch_acc_1=100*running_corrects_1/len(image_datasets[phase])
        epoch_acc_2=100*running_corrects_2/len(image_datasets[phase])
        print("{} Model1 Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss_1,epoch_acc_1))
        print("{} Model2 Loss:{:.4f} Acc:{:.4f}%".format(phase,epoch_loss_2,epoch_acc_2))
    time_Dur=time.time()-time_start
    print(time_Dur)