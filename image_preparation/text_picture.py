        
import numpy as np
from PIL import Image
import scipy.ndimage as pyimg
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch
import random       
        

def text_mask(filename_mask,filename_img):    

        A_img = Image.open(filename_img).convert('RGB')
        A_mask= Image.open(filename_mask).convert('RGB')
        w, h = A_mask.size
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
        return A_img


# mask path 
#bin_text=  "/home/maowendong/project/style_transfer/text_nonsyn2/inp_preprocess/text_mask/Q_smooth.jpg"  #text_mask
bin_text = "/home/maowendong/project/style_transfer/text_nonsyn2/inp_preprocess/text_mask/lan.jpg"
# cut picture by mask
style_img_path=  "/home/maowendong/project/style_transfer/text_nonsyn2/inp_preprocess/taohua2.jpg"  #texture_path
imag = text_mask(bin_text, style_img_path )

#texture_t_path   
imag.save("tree2.jpg")