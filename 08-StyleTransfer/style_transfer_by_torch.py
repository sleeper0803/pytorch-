import __future__
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
内容损失函数
content_feature是通过卷积获取到的输入图像中的内容
content_feature.detach()用于对提取到的内容进行锁定，不需要进行梯度
weight是设置的权重参数，用来控制内容和风格对最后合成图像的影响程度
forward函数用于计算输入图像和内容图像之间的损失值
backward函数根据计算得到的损失值进行后向传播，并返回损失值
'''
class ContentLoss(torch.nn.Module):
    def __init__(self,content_feature,weight):
        super(ContentLoss,self).__init__()
        self.content_feature = content_feature.detach()
        self.criterion = torch.nn.MSELoss()
        self.weight = weight

    def forward(self,combination):
        self.loss = self.criterion(combination.clone()*self.weight,self.content_feature.clone()*self.weight)
        return combination

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss
'''
风格损失函数
代码基本和内容损失函数相似，引入格拉姆矩阵参与风格损失的计算
格拉姆矩阵（GramMatrix）
通过卷积神经网络提取风格图片的风格，风格由数字组成，数字的大小代表了图片中风格的突出程度，
Gram矩阵是矩阵的内积计算，在运算过后输入到该矩阵的特征图中的大的数字会变得更大，相当于图片的风格被放大了，
放大后的风格再参与损失计算，便能够对最后的合成图片产生更大的影响
https://baike.baidu.com/item/%E6%A0%BC%E6%8B%89%E5%A7%86%E7%9F%A9%E9%98%B5/16274086?fr=aladdin
'''
class GramMatrix(torch.nn.Module):
    def forward(self, input):
        b, n, h, w = input.size()  
        features = input.view(b * n, h * w) 
        G = torch.mm(features, features.t()) 
        return G.div(b * n * h * w)

class StyleLoss(torch.nn.Module):
    def __init__(self,style_feature,weight):
        super(StyleLoss,self).__init__()
        self.style_feature = style_feature.detach()
        self.criterion = torch.nn.MSELoss()
        self.weight = weight
        self.gram = GramMatrix()

    def forward(self,combination):
        #output = combination
        style_feature = self.gram(self.style_feature.clone()*self.weight)
        combination_features = self.gram(combination.clone()*self.weight)
        self.loss = self.criterion(combination_features,style_feature)
        return combination

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

class StyleTransfer:
    def __init__(self,content_image,style_image,style_weight=1000,content_weight=1):
        # Weights of the different loss components
        self.vgg19 =  models.vgg19()
        self.vgg19.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'), strict=True)
        self.img_ncols = 400
        self.img_nrows = 300
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.content_tensor,self.content_name = self.process_img(content_image)
        self.style_tensor,self.style_name = self.process_img(style_image)
        self.conbination_tensor = self.content_tensor.clone()

    def process_img(self,img_path):
        img = Image.open(img_path)
        img_name  = img_path.split('/')[-1][:-4]
        loader = transforms.Compose([transforms.Resize((self.img_nrows,self.img_ncols)),
        transforms.ToTensor()])
        img_tensor = loader(img)
        #plt.imshow(img)
        #plt.show()
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(device, torch.float),img_name
    
    def deprocess_img(self,x,index):
        unloader = transforms.ToPILImage()
        x = x.cpu().clone()
        img_tensor = x.squeeze(0)
        img = unloader(img_tensor)
        result_folder = f'{self.content_name}_and_{self.style_name}'
        os.path.exists(result_folder) or os.mkdir(result_folder)
        filename = f'{result_folder}/rersult_{index}.png'
        img.save(filename)
        print(f'save {filename} successfully!')
        print()

    def get_loss_and_model(self,vgg_model,content_image,style_image):
        vgg_layers = vgg_model.features.to(device).eval()
        
        style_losses = []
        content_losses = []
        model = torch.nn.Sequential()
        style_layer_name_maping = {
                '0':"style_loss_1",
                '5':"style_loss_2",
                '10':"style_loss_3",
                '19':"style_loss_4",
                '28':"style_loss_5",
            }
        content_layer_name_maping = {'30':"content_loss"}
        for name,module in vgg_layers._modules.items():
            model.add_module(name,module)
            if name in content_layer_name_maping:
                content_feature = model(content_image).clone()
                content_loss = ContentLoss(content_feature,self.content_weight)
                model.add_module(f'{content_layer_name_maping[name]}',content_loss)
                content_losses.append(content_loss)
            if name in style_layer_name_maping:
                style_feature = model(style_image).clone()
                style_loss = StyleLoss(style_feature,self.style_weight)
                style_losses.append(style_loss)
                model.add_module(f'{style_layer_name_maping[name]}',style_loss)
        #print(model)
        #content_losses和style_losses是两个用于保存内容损失函数和风格损失的列表
        return content_losses,style_losses,model

    def get_input_param_optimizer(self,input_img):
        #可以把torch.nn.Parameter理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        #并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        #所以在参数优化的时候可以进行优化的)，所以经过类型转换这个input_param变成了模型的一部分，
        #成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        #https://www.jianshu.com/p/d8b77cc02410
        input_param = torch.nn.Parameter(input_img.data)
        optimizer = torch.optim.LBFGS([input_param])
        return input_param, optimizer

    def main_train(self,epoch=10):
        combination_param, optimizer = self.get_input_param_optimizer(self.conbination_tensor)
        content_losses,style_losses,model = self.get_loss_and_model(self.vgg19,self.content_tensor,self.style_tensor)
        cur,pre = 1e20,1e20
        for i in range(1,epoch+1):
            start = time.time()
            def closure():
                combination_param.data.clamp_(0,1)
                optimizer.zero_grad()
                model(combination_param)
                style_score = 0
                content_score = 0
                for cl in content_losses:
                    content_score += cl.loss
                for sl in style_losses:
                    style_score += sl.loss
                loss =  content_score+style_score
                loss.backward()
                if i % 10==0:
                    print('Epoch:{} Style Loss: {:4f} Content Loss: {:4f}'.format(i,style_score,content_score))
                return style_score+content_score
            loss = optimizer.step(closure)
            cur,pre = loss,cur
            end = time.time()
            print(f'|using:{int(end-start):2d}s |epoch:{i:2d} |loss:{loss.data}')

            if pre<=cur:
                print('Early stopping!')
                break
            combination_param.data.clamp_(0,1)
            if i%5 == 0:
                self.deprocess_img(self.conbination_tensor,i//5)
            
if __name__ == "__main__":
    pass
    print('welcome')
    content_file = 'images/Taipei101.jpg'
    style_file = 'images/Feathers.jpg'
    st = StyleTransfer(content_file,style_file)
    epoch = 100
    st.main_train(epoch=epoch)