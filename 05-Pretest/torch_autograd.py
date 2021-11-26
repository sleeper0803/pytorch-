import torch
from torch.autograd import Variable


#构建神经网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        print(50)
        #super(Model,Self).__init__()
    def forward(self,input,w1,w2):
        x=torch.mm(input,w1)
        x=torch.clamp(x,min=0)
        x=torch.mm(x,w2)
        return x
    
    def backward(self):
        pass


batch_n = 100 # 批量输入的数据量
hidden_layer = 100 # 通过隐藏层后输出的特征数
input_data = 1000 # 输入数据的特征个数
output_data = 10 # 最后输出的分类结果数

#初始化权重
x=Variable(torch.randn(batch_n,input_data),requires_grad = False)
y=Variable(torch.randn(batch_n,output_data),requires_grad = False)
w1=Variable(torch.randn(input_data,hidden_layer),requires_grad = True)
w2=Variable(torch.randn(hidden_layer,output_data),requires_grad = True)

epoch_n = 20  #训练的次数
learning_rate = 1e-6 #学习效率


model=Model() #对模型类进行调用
for epoch in range(epoch_n):
    
    y_pred = model(x,w1,w2) #完成对模型预测值的输出
    y_pred2=x.mm(w1).clamp(min=0).mm(w2)
    
    loss=(y_pred-y).pow(2).sum()
    print("epoch:{},Loss:{:,.4f}".format(epoch,loss.data))
    #print(epoch,"{:.4f}".format(loss.item()))
    loss.backward()
    
    with torch.no_grad():
        w1.data-=learning_rate*w1.grad.data;
        w2.data-=learning_rate*w2.grad.data;

        w1.grad.data.zero_()
        w2.grad.data.zero_()