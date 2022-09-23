import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.nn.modules.utils import _pair, _triple
import torch.nn.functional as F
from torch.nn.modules import conv, Linear
import numpy as np
from torchvision import transforms
from Structure_Net import utils
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from PIL import Image
import PIL
import random
import cv2 
from pro_gen_GAN.models import networks
import argparse
 

img_transform = transforms.Compose([
            #transforms.Scale(320),                  # scale shortest side to image_size
            #transforms.CenterCrop(320),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            #utils.normalize_tensor_transform()      # normalize with ImageNet values
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])


def image_preprocess_remove(filename):
    Iw = Image.open(filename)
    x,y = Iw.size
    #Iw =Iw.crop((0,0,y,y))
    Iw = np.array(Iw) 
    
    BW1=((Iw[:,:,0]>100)&( Iw[:,:,1]<20)&(Iw[:,:,2]<20) )  #white
    #BW1=((Iw[:,:,0]>200)&( Iw[:,:,1]>200)&(Iw[:,:,2]>200) )  #white
    channelw0=Iw[:,:,0]
    channelw0[BW1]=255
    channelw1=Iw[:,:,1]
    channelw1[BW1]=255
    channelw2=Iw[:,:,2]
    channelw2[BW1]=255
    
    BW1=((Iw[:,:,0]<100)&( Iw[:,:,1]<100)&(Iw[:,:,2]<200) ) #black
    #BW1=((Iw[:,:,0]<254)&( Iw[:,:,1]<254)&(Iw[:,:,2]<254) )  #white
    channelw0=Iw[:,:,0]
    channelw0[BW1]=0
    channelw1=Iw[:,:,1]
    channelw1[BW1]=0
    channelw2=Iw[:,:,2]
    channelw2[BW1]=0
    
    Iw[:,:,0]=channelw0.astype('uint8')
    Iw[:,:,1]=channelw1.astype('uint8')
    Iw[:,:,2]=channelw2.astype('uint8')
    #Iw[:,:,:]=255-Iw[:,:,:]   # modify for text mask
    Iw=Image.fromarray(Iw)
    return Iw


def controllable_mask(real_B, deform_level):
        real_B = cv2.cvtColor(np.array(real_B),cv2.COLOR_BGR2RGB)
        if( deform_level == 0 ):  # deform_level 0,1,2,3
            kernel_size = 1
            center = 0
        elif( deform_level == 1 ):
            kernel_size = 3
            center = 2
        elif( deform_level == 2 ):
            kernel_size = 7        #9
            center = 6
        else:
            kernel_size = 9   
            center = 10
        
        gauss = cv2.getGaussianKernel(kernel_size, 0)
        gauss = gauss * gauss.transpose(1, 0)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if (deform_level > 0):
            k = np.ones((3, 3), np.uint8)
            real_B =  cv2.dilate(real_B, k, iterations = deform_level )
        #k = np.ones((3, 3), np.uint8)    
        #real_B =  cv2.dilate(real_B, k, iterations = 2 )    
        edges = cv2.Canny(real_B,100,200)
        dilation = cv2.dilate(edges, kernel)
        gauss_img = np.copy(real_B)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
                x = random.randint(0,15)
                if x <1 :
                    m = 0
                else:
                    m = 255
                gauss_img[idx[0][i], idx[1][i], 0] = m
                gauss_img[idx[0][i], idx[1][i], 1] = m
                gauss_img[idx[0][i], idx[1][i], 2] = m
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gauss_img = cv2.erode(gauss_img, kernel2, iterations=3)
        if (deform_level>1):
            nc = 2
        else:
            nc = 1

        gauss_img = cv2.dilate(gauss_img, kernel2, iterations=1)
        (_, thresh) = cv2.threshold(gauss_img, 60, 255, cv2.THRESH_BINARY)
        mask_img = 255 - np.copy(thresh)

        if 0 < center<10:
            center_control = (10 - center)*5
        elif center>9:
            center_control = 2
        else:
            center_control = 10000

        print('The deformable scale is' + ' ' + str(deform_level)+ '!\n' )
        idx = np.where(mask_img != 0)
        for i in range(np.sum(mask_img != 0)):
                    x = random.randint(0, center_control)
                    if x < 1 :
                        m = 255
                    else:
                        m = 0
                    gauss_img[idx[0][i], idx[1][i], 0] = m
                    gauss_img[idx[0][i], idx[1][i], 1] = m
                    gauss_img[idx[0][i], idx[1][i], 2] = m

        (_, gauss_img) = cv2.threshold(gauss_img, 60, 255, cv2.THRESH_BINARY)
        gauss_img = Image.fromarray(cv2.cvtColor(gauss_img,cv2.COLOR_BGR2RGB))
        return gauss_img



def text_mask(filename_wholeimg, text_mask_path, deform_level):    

        A_mask = image_preprocess_remove(text_mask_path)
        A_mask = controllable_mask(A_mask, deform_level)

        crop_size = 320              #  modify crop size for different patch
        mask_size = 320
        A_img = Image.open(filename_wholeimg).convert('RGB')
        w_Aimg, h_Aimg = A_img.size
        rw = random.randint(0, w_Aimg - crop_size)
        rh = random.randint(0, h_Aimg - crop_size)     
        #rw = 8
        #rh =8
        #A_img = A_img.crop((rw, rh, rw + crop_size, rh + crop_size))
        #A_mask= Image.open(filename_mask).convert('RGB')
        w1, h1 = A_mask.size
        A_mask=A_mask.resize((mask_size,int(h1*mask_size/w1)),Image.BILINEAR) 

        w, h = A_mask.size    
        A_img = A_img.crop((rw, rh, rw + w, rh + h))
        A_mask = A_mask.crop((0,0,w-0,h-0)).resize((w,h),Image.BILINEAR)
        A_img=A_img.resize((w,h),Image.BILINEAR)
        IA = np.array(A_mask)
        npA_img=np.array(A_img)
        
        BW1=((IA[:,:,0]>50)&( IA[:,:,1]>50)&(IA[:,:,2]>50) )
        channel0=npA_img[:,:,0]
        channel0[BW1]=255
        channel1=npA_img[:,:,1]
        channel1[BW1]=255
        channel2=npA_img[:,:,2]
        channel2[BW1]=255
        npA_img[:,:,0]=channel0.astype('uint8')
        npA_img[:,:,1]=channel1.astype('uint8') 
        npA_img[:,:,2]=channel2.astype('uint8')
    
        A_img=Image.fromarray(npA_img)
        return A_mask,A_img


def get_prototype( model_pretrain_path, pic_path, text_mask_path, deform_level):
        netG = networks.UnetGenerator(input_nc=3, output_nc=3,num_downs=1, ngf=64, norm_layer= nn.BatchNorm2d, use_dropout=True,
                        gpu_ids=[])

        A_mask,A_img = text_mask(pic_path, text_mask_path, deform_level)
        A_img = img_transform(A_img)
        A_mask = img_transform(A_mask)
        inp= torch.cat((A_img,A_mask), 0)  

        B_tensor = inp
        dtype = torch.cuda.FloatTensor
        inp = Variable(inp.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

        # load style model
        texture_model =netG.type(dtype)
        texture_model.cuda()

        state_dict =torch.load(model_pretrain_path)
        texture_model.load_state_dict(state_dict)
        # process input image
        texture_out = texture_model(inp).cpu()
        texture_out = (texture_out + 1 )/2
        texture_out = transforms.ToPILImage()(texture_out[0])
        return texture_out


def get_segmask(path_gan_out, model_pretrain_path ):
        netG_mask = networks.SegGenerator(input_nc=3, output_nc=3,num_downs=1, ngf=64, norm_layer= nn.BatchNorm2d, use_dropout=True,
                        gpu_ids=[])

        A_img = Image.open(path_gan_out).convert('RGB')
        inp = img_transform(A_img)
        dtype = torch.cuda.FloatTensor
        inp = Variable(inp.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

        # load style model
        mask_model =netG_mask.type(dtype)
        mask_model.cuda()

        state_dict =torch.load(model_pretrain_path)

        mask_model.load_state_dict(state_dict)
        output_path = 'Gp2_segmask.jpg'
        # process input image
        mask_out = mask_model(inp).cpu()
        mask_out = (mask_out + 1 )/2
        mask_out = transforms.ToPILImage()(mask_out[0])
        mask_out.save(output_path)
        return mask_out


def binary_text(filename_textimg,gan_out_path,mask):
    Iw = Image.open(gan_out_path).convert('RGB')
    x,y = Iw.size
    #Iw =Iw.crop((0,0,y,y))
    Iw = np.array(Iw) 
    np_Iw = Iw.copy()

    A_img = Image.open(filename_textimg).convert('RGB')
    IA = np.array(mask)
    npA_img=np.array(A_img)
    BW1=((IA[:,:,0]>50)&( IA[:,:,1]>50)&(IA[:,:,2]>50) )
    channel0=npA_img[:,:,0]
    np_Iw0=np_Iw[:,:,0]
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
    return A_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Gp1path", type=str, default="./pro_gen_GAN/checkpoints/leaf_nocas/honghua_15000_net_G.pth", help="path to a pretrain model of Gp1")
    parser.add_argument("--picpath", type=str, default="./pro_gen_GAN/inp_preprocess/test_case/202honghua/train/2.jpg", help="path to a style image")
    parser.add_argument("--tpath", type=str, default= "./inp_preprocess/text_mask/love.jpg", help="path to an input text mask")
    parser.add_argument("--Gp2path", type=str, default="./pro_gen_GAN/checkpoints/Gp2/honghua_850_net_G.pth", help="path to a pretrain model of Gp2")
    parser.add_argument("--deforml", type=int, default=0, help="The deformable level of style transfer")
    args = parser.parse_args() 

    print('Stylized Text Protptype Generation!')
    prototype = get_prototype(args.Gp1path,args.picpath, args.tpath, args.deforml)
    path_Gp1_out= 'Gp1_prototype.jpg'
    prototype.save(path_Gp1_out)
    if args.Gp2path == 'no_path':
        pass
        #print('Alternative Transferring')
    else:    
        mask_out = get_segmask(path_Gp1_out, args.Gp2path)


if __name__ == '__main__':
    main()

















