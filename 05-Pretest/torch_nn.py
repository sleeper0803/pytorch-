import torch
from torch._C import ThroughputBenchmark
from torch.autograd import Variable

batch_n = 100 # 批量输入的数据量
hidden_layer = 100 # 通过隐藏层后输出的特征数
input_data = 1000 # 输入数据的特征个数
output_data = 10 # 最后输出的分类结果数

#初始化权重
x=Variable(torch.randn(batch_n,input_data),requires_grad=False)
y=Variable(torch.randn(batch_n,output_data),requires_grad=False)
#w1=Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
#w2=Variable(torch.randn(hidden_layer,output_data),requires_grad=True)

#搭建模型
models= torch.nn.Sequential(  
    # 首先通过其完成从输入层到隐藏层的线性变换
    torch.nn.Linear(input_data,hidden_layer),
    # 经过激活函数
    torch.nn.ReLU(),
    # 最后完成从隐藏层到输出层的线性变换
    torch.nn.Linear(hidden_layer,output_data))
print(models)

epoch_n=100
learning_rate=1e-4
loss_fn=torch.nn.MSELoss()

'''
for epoch in range(epoch_n):
    y_pred =models(x)
    loss= loss_fn(y_pred,y)
    if epoch % 1000 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch, loss.data))
    models.zero_grad()
    loss.backward()
    # 访问模型中的全部参数是通过对“models.parameters()”进行遍历完成的
    # 对每个遍历的参数进行梯度更新
    for param in models.parameters():
        param.data-=param.grad.data*learning_rate        
'''


optimzer=torch.optim.Adam(models.parameters(),lr=learning_rate) 
 # 训练模型
for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    print("Epoch:{},Loss:{:.4f}".format(epoch, loss.data))
    optimzer.zero_grad() # 参数梯度的归零
    loss.backward()
    optimzer.step() # 进行梯度更新

