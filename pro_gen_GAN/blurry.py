import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d, ReplicationPad2d
import torch.nn.functional as F
import random
import numpy as np
import math
from torchvision import transforms
import torch.autograd as autograd
from torch.autograd import Variable
import os
from PIL import Image

class myBlur(nn.Module):
    def __init__(self, kernel_size=13, channels=3):
        super(myBlur, self).__init__()
        kernel_size = int(int(kernel_size/2)*2)+1
        self.kernel_size=kernel_size
        self.channels = channels
        self.GF = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)
        x_cord = torch.arange(self.kernel_size+0.)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1)//2
        self.diff = -torch.sum((self.xy_grid - self.mean)**2., dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False)

        self.gaussian_filter.weight.requires_grad = False
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, sigma, gpu):
        sigma = sigma * 8. + 16.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(self.diff /(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        if gpu:
            gaussian_kernel = gaussian_kernel.cuda()
        self.gaussian_filter.weight.data = gaussian_kernel
        out = self.gaussian_filter(F.pad(x, (self.mean,self.mean,self.mean,self.mean), "replicate"))
        # out = self.sigmoid(out)
        return out

def save_image(filename, data):
    # std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    # mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = (img.transpose(1, 2, 0)*255.0).clip(0, 255)-125
    img = torch.from_numpy(img)
    sigmoid = nn.Sigmoid()
    img = sigmoid(img)*255
    img = img.numpy().astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)



def image_preprocess_remove(filename):
    Iw = np.array(Image.open(filename)) 
    
    BW1=((Iw[:,:,0]>190) )  #white
    channelw0=Iw[:,:,0]
    channelw0[BW1]=255
    channelw1=Iw[:,:,1]
    channelw1[BW1]=255
    channelw2=Iw[:,:,2]
    channelw2[BW1]=255
    
    
    BW1=((Iw[:,:,0]<250) ) #black
    channelw0=Iw[:,:,0]
    channelw0[BW1]=0
    channelw1=Iw[:,:,1]
    channelw1[BW1]=0
    channelw2=Iw[:,:,2]
    channelw2[BW1]=0
    
    Iw[:,:,0]=channelw0.astype('uint8')
    Iw[:,:,1]=channelw1.astype('uint8')
    Iw[:,:,2]=channelw2.astype('uint8')
    #Iw[:,:,:]=255-Iw[:,:,:]
    Iw=Image.fromarray(Iw)
    return Iw


#binary
name = 'cloth'
x1= '/home/maowendong/project/style_transfer/text_nonsyn2/inp_preprocess/style_img/'+ name +'.jpg'  #source photos
imag = image_preprocess_remove(x1)
imag.save("bin_mask1.jpg")  #bin_mask



#blur
# load input image
input_img = Image.open("bin_mask1.jpg") 
totensor = transforms.ToTensor()
input_img = totensor(input_img)
input_img = Variable(input_img.repeat(1, 1, 1, 1), requires_grad=False)
# create model
blurry = myBlur()
output_img = blurry(x = input_img, sigma = 0.1, gpu = False)
# save image
output_path = 'blurry_mask/blurry_' + name + 'mask.jpg' 
save_image(output_path, output_img.data[0])

    