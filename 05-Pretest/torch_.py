# -*- coding: utf-8 -*-

import torch
batch_n=100
hidden_layer=100
input_data=1000
output_data=10

x=torch.randn(batch_n,input_data)
y=torch.randn(batch_n,output_data)
w1=torch.randn(input_data,hidden_layer)
w2=torch.randn(hidden_layer,output_data)

print(x)


epoch_n=20
learning_rate=1e-6

for epoch in range(epoch_n):
    h1=x.mm(w1)
    #print(h1)
    h1=h1.clamp(min=0)
    y_pred=h1.mm(w2)
    
    loss=(y_pred-y).pow(2).sum()
    print("epoch:{},Loss:{:,.1f}".format(epoch,loss))
    
    grad_y_pred=2*(y_pred-y)
    grad_w2=h1.t().mm(grad_y_pred)
    
    grad_h=grad_y_pred.clone()
    grad_h=grad_h.mm(w2.t())
    grad_h.clamp(min=0)
    grad_w1=x.t().mm(grad_h)
    
    w1-=learning_rate*grad_w1
    w2-=learning_rate*grad_w2
    
    #grad_y_pred=2(x*w1*w2-y)
    #w1=w1-2*xT*(x*w1*w2-y)*W2T*learning_rate
    #w2=w2-2(x*w1)T(x*w1*w2-y)*learning_rate

    print(w1)
    print(w2)

