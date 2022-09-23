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
import shutil  
import os 
import cv2

class myBlur(nn.Module):
    def __init__(self, kernel_size=13, channels=3):  #doushi 19
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
    #img = 255 - img  # modify
    img = img.numpy().astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)



def image_preprocess_remove(filename):
    Iw = np.array(Image.open(filename)) 
    
    BW1=((Iw[:,:,0]>240) )  #white
    channelw0=Iw[:,:,0]
    channelw0[BW1]=255
    channelw1=Iw[:,:,1]
    channelw1[BW1]=255
    channelw2=Iw[:,:,2]
    channelw2[BW1]=255
    
    
    BW1=((Iw[:,:,0]<240) ) #black
    channelw0=Iw[:,:,0]
    channelw0[BW1]=0
    channelw1=Iw[:,:,1]
    channelw1[BW1]=0
    channelw2=Iw[:,:,2]
    channelw2[BW1]=0
    
    Iw[:,:,0]=channelw0.astype('uint8')
    Iw[:,:,1]=channelw1.astype('uint8')
    Iw[:,:,2]=channelw2.astype('uint8')
    #Iw[:,:,:]=255-Iw[:,:,:]  #modify
    Iw=Image.fromarray(Iw)
    return Iw


def text_mask(filename_mask,filename_img):    

        A_img = Image.open(filename_img).convert('RGB')
        A_mask= Image.open(filename_mask).convert('RGB')
        w, h = A_mask.size
        A_img=A_img.resize((w,h),Image.BILINEAR)
        IA = np.array(A_mask)
        npA_img=np.array(A_img)
        BW1=((IA[:,:,0]>100)&( IA[:,:,1]>100)&(IA[:,:,2]>100) )
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





if __name__ == '__main__':
    name = 'honghua'

    
    folder = './test_case/202'+ name
    if os.path.exists(folder):
        pass
    else:
        os.mkdir(folder)

    folder_train = './test_case/202'+ name + '/train'    
    if os.path.exists(folder_train):
        pass
    else:
        os.mkdir(folder_train)

    folder_test = './test_case/202'+ name + '/test'    
    if os.path.exists(folder_test):
        pass
    else:
        os.mkdir(folder_test)

    folder_label = './test_case/202'+ name + '/label'    
    if os.path.exists(folder_label):
        pass
    else:
        os.mkdir(folder_label)


    
    for i in range(1,3):
            x1= 'style_img/'+ name +str(i)+'.jpg'  #source photos
            imag = image_preprocess_remove(x1)
            path_bin = 'mask'+ str(i)+'_O.jpg' 
            imag.save(path_bin)  #bin_mask
            pic_path = folder_train +'/' + str(i) + '.jpg'
            mask_path = folder +'/mask' + str(i) + '.jpg'
            mask_bin_path = folder +'/mask' + str(i) + '_O.jpg' 
            # load input image
            
            ''' 
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            imag = cv2.cvtColor(np.array(imag),cv2.COLOR_BGR2RGB)
            #imag = cv2.erode(imag, kernel2, iterations=2)
            imag = cv2.dilate(imag, kernel2, iterations=2)  #bianxiao
            '''
            input_img = imag
            totensor = transforms.ToTensor()
            input_img = totensor(input_img)
            input_img =  Variable(input_img.repeat(1, 1, 1, 1), requires_grad=False)
            # create model
            blurry = myBlur()
            for i in range(2):
                 input_img = blurry(x = input_img, sigma = 0.1, gpu = False)
            # save image
            output_path = 'blurry_mask/' + name + str(i)+ '_blurry_mask.jpg' 
            save_image(output_path, input_img.data[0])

            shutil.copyfile(x1, pic_path)
            shutil.copyfile(output_path, mask_path)
            shutil.copyfile( path_bin, mask_bin_path)

    # mask path 
    bin_text=  "text_mask/E.jpg"  #text_mask

    # cut picture by mask
    style_img_path=  'style_img/'+ name + '2.jpg'   #texture_path
    imag = text_mask(bin_text, style_img_path )
    #texture_t_path 
    texture_t_path = "text.jpg"
    imag.save( texture_t_path ) 


    label_path = folder_label + '/dismap.jpg'
    test_texture_path = folder_test  + "/E" + name +".jpg"
    shutil.copyfile(bin_text, label_path)
    shutil.copyfile(texture_t_path, test_texture_path)