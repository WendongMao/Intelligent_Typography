import __future__
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ContentLoss(torch.nn.Module):
    def __init__(self,content_feature,weight):
        super(ContentLoss,self).__init__()
        self.content_feature = content_feature.detach()
        self.criterion = torch.nn.MSELoss()
        self.weight = weight

    def forward(self,combination):
        self.loss = self.criterion(combination.clone()*self.weight,self.content_feature.clone()*self.weight)
        return combination

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
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = torch.nn.MSELoss()

    def forward(self,combination):
        #output = combination
        style_feature = self.gram(self.style_feature.clone()*self.weight)
        combination_features = self.gram(combination.clone()*self.weight)
        self.loss = self.criterion(combination_features,style_feature)
        return combination






class StyleTransfer:
    def __init__(self,content_image,style_image,style_weight=18,content_weight=0.000):
        # Weights of the different loss components
        self.vgg19 =  models.vgg19()
        self.vgg19.load_state_dict(torch.load("./Texture_Net/vgg19-dcbb9e9d.pth"))
        self.img_ncols = 400
        self.img_nrows = 300
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.content_tensor,self.content_name = self.process_img2(content_image)
        self.style_tensor,self.style_name = self.process_img1(style_image)
        self.conbination_tensor = self.content_tensor.clone()

    def process_img1(self,img_path):
        img = Image.open(img_path)
        self.img_ncols, self.img_nrows = img.size
        img = img.crop((self.img_ncols/2-325,self.img_nrows/2-325,self.img_ncols/2+325,self.img_nrows/2+325))
        img_name  = img_path.split('/')[-1][:-4]
        loader = transforms.Compose([transforms.Resize((self.img_nrows,self.img_ncols)),
        transforms.ToTensor()])
        img_tensor = loader(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(device, torch.float),img_name

    def process_img2(self,img_path):
        img = Image.open(img_path)
        w,h = img.size
        img = np.array(img.resize((w+2,h+2),Image.BILINEAR))
        img = Image.fromarray(self.smooth(img))
        self.img_ncols, self.img_nrows = img.size
        img_name  = img_path.split('/')[-1][:-4]
        loader = transforms.Compose([transforms.Resize((self.img_nrows,self.img_ncols)),
        transforms.ToTensor()])
        img_tensor = loader(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(device, torch.float),img_name


    def binary_text(self,inp,gan_out_path,mask_path):
        unloader = transforms.ToPILImage()
        inp = inp.cpu().clone()
        inp = inp.squeeze(0)
        A_img = unloader(inp)
        if mask_path == 'no_path':
            pass
        else:
            mask = Image.open(mask_path).convert('RGB')
            Iw = Image.open(gan_out_path).convert('RGB')
            x,y = Iw.size
            Iw = np.array(Iw) 
            np_Iw = Iw.copy()
            IA = np.array(mask)
            npA_img=np.array(A_img)
            BW1=((IA[:,:,0]>100)&( IA[:,:,1]>100)&(IA[:,:,2]>100) )
            channel0=npA_img[:,:,0]
            np_Iw0= np_Iw[:,:,0]
            channel0[BW1]= np_Iw0[BW1]     #255 for white back ground
            channel1=npA_img[:,:,1]
            np_Iw1=np_Iw[:,:,1]
            channel1[BW1]= np_Iw1[BW1]
            channel2=npA_img[:,:,2]
            np_Iw2=np_Iw[:,:,2]
            channel2[BW1]= np_Iw2[BW1] 
            npA_img[:,:,0]=channel0.astype('uint8')
            npA_img[:,:,1]=channel1.astype('uint8') 
            npA_img[:,:,2]=channel2.astype('uint8')
            A_img=Image.fromarray(npA_img)
        loader = transforms.Compose([transforms.ToTensor()])
        img_tensor = loader(A_img)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor


    def deprocess_img(self,x,index):
        #x= self.deprocess_image1(x)
        unloader = transforms.ToPILImage()
        x = x.cpu().clone()
        img_tensor = x.squeeze(0)
        img = unloader(img_tensor)

        image_enhance = True
        if image_enhance:
            # brightness enhance
            enh_bri = ImageEnhance.Brightness(img)
            brightness = 1.05
            img = enh_bri.enhance(brightness)

            # chroma enhance
            enh_col = ImageEnhance.Color(img)
            color =  0.95
            img = enh_col.enhance(color)

            # acutance enhance
            enh_sha = ImageEnhance.Sharpness(img)
            sharpness = 0.95
            img = enh_sha.enhance(sharpness)

        #img.save(filename)
        #print(f'save {filename} successfully!')
        if index == 1:
           img.save('./Nt_results.jpg')
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
                if name =='0':
                   self.style_weight = 6
                else:
                   self.style_weight = 4
                style_loss = StyleLoss(style_feature,self.style_weight)
                style_losses.append(style_loss)
                model.add_module(f'{style_layer_name_maping[name]}',style_loss)
        return content_losses,style_losses,model

    def get_input_param_optimizer(self,input_img):
        input_param = torch.nn.Parameter(input_img.data)
        optimizer = torch.optim.LBFGS([input_param])
        return input_param, optimizer

    
    def smooth(self, image):  # 模糊图片
        w, h,c = image.shape
        smoothed_image = np.zeros([w - 2, h - 2,c])
        smoothed_image += image[:w - 2, 2:h,:]
        smoothed_image += image[1:w-1, 2:,:]
        smoothed_image += image[2:, 2:h,:]
        smoothed_image += image[:w-2, 1:h-1,:]
        smoothed_image += image[1:w-1, 2:h,:]
        smoothed_image += image[2:, 1:h - 1,:]
        smoothed_image += image[:w-2, :h-2,:]
        smoothed_image += image[1:w-1, :h - 2,:]
        smoothed_image += image[2:, :h - 2,:]
        smoothed_image /= 9.0
        #print(smoothed_image.shape)
        return smoothed_image.astype("uint8")
    


    def main_train(self,epoch, mask_path, gan_out_path):
        combination_param, optimizer = self.get_input_param_optimizer(self.conbination_tensor)
        content_losses,style_losses,model = self.get_loss_and_model(self.vgg19,self.content_tensor,self.style_tensor)
        cur,pre = 10,10
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
                return style_score+content_score
            loss = optimizer.step(closure)
            cur,pre = loss,cur
            end = time.time()
            print(f'|using:{int(end-start):2d}s |epoch:{i:2d} |loss:{loss.data}')

            combination_param.data.clamp_(0,1)

            if i%1 == 0:
                self.conbination_tensor = self.binary_text(self.conbination_tensor,gan_out_path,mask_path)
                self.deprocess_img(self.conbination_tensor,i)
                
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--picpath", type=str, default="./pro_gen_GAN/inp_preprocess/test_case/202honghua/train/1.jpg", help="path to a style image")
    parser.add_argument("--Gp1opath", type=str, default="./Gp1_prototype.jpg", help="path to a stylied text prototype")
    parser.add_argument("--Gp2opath", type=str, default="./Gp2_segmask.jpg", help="path to a Gp2 results")
    parser.add_argument("--Nsopath", type=str, default= "./Ns_result.jpg", help="path to a sturcture refined results")
    args = parser.parse_args() 
    start = time.time()
    print('Forward Texture Refinement!')
    content_file = args.Nsopath
    style_file = args.picpath
    st = StyleTransfer(content_file,style_file)
    epoch = 2
    mask_path = args.Gp2opath
    gan_out_path = args.Gp1opath
    st.main_train(epoch,mask_path, gan_out_path)
    end = time.time()
    print(f'|using:{int(end-start):2d}s')

