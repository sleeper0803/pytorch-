#import torchvision
from torchvision_ import*
from torchvision_model import*
#import torch

n_epoches=5

model=Model()
cost=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())

#训练
for epoch in range (n_epoches):
    running_loss=0.0
    running_correct=0
    print("Epoch{}/{}".format(epoch,n_epoches))
    print("*"*50)
    for data in  data_loader_train :
        X_train,Y_train=data
        X_train,Y_train=Variable(X_train),Variable(Y_train)
        outputs=model(X_train)
        _,pred=torch.max(outputs.data,1)
        optimizer.zero_grad()
        loss=cost(outputs,Y_train)

        loss.backward()
        optimizer.step()
        running_loss+=loss.data
        running_correct+=torch.sum(pred==Y_train.data)
    testing_correct=0
    for data in data_loader_test:
        X_test,Y_test=data
        X_test,Y_test=Variable(X_test),Variable(Y_test)
        outputs=model(X_test)
        _,pred=torch.max(outputs.data,1)
        testing_correct+=torch.sum(pred==Y_test.data)
    print("Loss is :{:.4f},Train Accuracy is : {:.4f}%,Test Accuracy is :{:.4f}"
                .format(running_loss/len(data_train),
                100*running_correct/len(data_train),
                100*testing_correct/len(data_test)))

data_loader_test2=torch.utils.data.DataLoader(dataset=data_test,
                        batch_size=4,shuffle=True)
X_test,Y_test=next(iter(data_loader_test2))
inputs=Variable(X_test)
pred=model(inputs)
_,pred=torch.max(pred,1)
print("Predict Label is :",[i for i in pred.data])
print("Real Label is : ",[i for i in Y_test])
img=utils.make_grid(X_test)
img=img.numpy().transpose(1,2,0)

std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
img=img*std+mean
plt.imshow(img)
pylab.show()


