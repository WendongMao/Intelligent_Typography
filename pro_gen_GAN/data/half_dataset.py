import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_half_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import PIL
from pdb import set_trace as st
import random
import torchvision.utils as vutils
import torch


class HalfDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.fineSize = opt.fineSize  #self.fineSize = 256
        self.transform = get_transform(opt)

    def __getitem__(self, index):
          choose = random.randint(0, 10)
          m = random.randint(-20, 20)
          index= 1
          if self.opt.which_model_netG == 'Gp_1':
               if (choose<1):
                    path_Bmask = "./datasets/half/202/mask1.jpg"
                    path = "./datasets/half/202/train/1.jpg"
                    B_img = Image.open(path).convert('RGB')
                    B_mask = Image.open(path_Bmask).convert('RGB')
                    if self.opt.isTrain and not self.opt.no_flip:
                         if random.random() > 0.5:
                              B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
                         else:
                              B_img = B_img

                    whole_image = B_img        
                    wB, hB = B_img.size
                    rw = random.randint(0, wB - self.fineSize-0)##
                    rh = random.randint(0, hB - self.fineSize-0)##
                    B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    B_mask = B_mask.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    w, h = B_img.size
                    A_mask=B_mask.resize((int(w/2),int(h/2)),Image.BILINEAR)
                    if(rw<int(wB/2)):
                         wa = int(rw + self.fineSize/4 +m)
                    else:
                         wa = int(rw + self.fineSize/4 -m)
                    if(rh<int(hB/2)):     
                         ha = int (rh + self.fineSize/4 +m)
                    else:
                         ha = int (rh + self.fineSize/4 -m) 
                    A_img=whole_image.crop((wa, ha, int(wa + self.fineSize/2), int(ha + self.fineSize/2)))
                    
                    #add mask for A_img
                    IA = np.array(A_mask)
                    npA_img=np.array(A_img)
                    BW1=((IA[:,:,0]>250)&( IA[:,:,1]>250)&(IA[:,:,2]>250) )
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
                    A_img = self.transform(A_img)
                    B_img = self.transform(B_img)
                    A_mask = self.transform(A_mask)
                    B_mask = self.transform(B_mask)
                    A_img= torch.cat((A_img,A_mask), 0)
               else:
                    path_Bmask = "./datasets/half/202/mask2.jpg"
                    path = "./datasets/half/202/train/2.jpg"
                    B_img = Image.open(path).convert('RGB')
                    B_mask = Image.open(path_Bmask).convert('RGB')
                    if self.opt.isTrain and not self.opt.no_flip:
                         if random.random() > 0.5:
                              B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
                         else:
                              B_img = B_img

                    whole_image = B_img        
                    wB, hB = B_img.size
                    rw = random.randint(0, wB - self.fineSize)##
                    rh = random.randint(0, hB - self.fineSize)##
                    B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    B_mask = B_mask.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    w, h = B_img.size
                    A_mask=B_mask.resize((int(w/2),int(h/2)),Image.BILINEAR)
                    if(rw<int(wB/2)):
                         wa = int(rw + self.fineSize/4 +m)
                    else:
                         wa = int(rw + self.fineSize/4 -m)
                    if(rh<int(hB/2)):     
                         ha = int (rh + self.fineSize/4 +m)
                    else:
                         ha = int (rh + self.fineSize/4 -m)     
                    A_img=whole_image.crop((wa, ha, int(wa + self.fineSize/2), int(ha + self.fineSize/2)))
                    
                    #add mask for A_img
                    IA = np.array(A_mask)
                    npA_img=np.array(A_img)
                    BW1=((IA[:,:,0]>250)&( IA[:,:,1]>250)&(IA[:,:,2]>250) )
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

                    A_img = self.transform(A_img)
                    B_img = self.transform(B_img)

                    A_mask = self.transform(A_mask)
                    B_mask = self.transform(B_mask)
                    A_img= torch.cat((A_img,A_mask), 0)   
          else:
               if (choose<1):
                    path_original_mask = "./datasets/half/202/mask1_O.jpg"
                    Bmask_O =  Image.open(path_original_mask).convert('RGB')
                    path = "./datasets/half/202/train/1.jpg"
                    B_img = Image.open(path).convert('RGB')      
                    wB, hB = B_img.size

                    w1 = random.randint(wB-100, wB +100)
                    h1 = random.randint(hB-100, hB +100)
                    B_img=B_img.resize((w1,h1),Image.BILINEAR)
                    Bmask_O =Bmask_O.resize((w1,h1),Image.BILINEAR)

                    rw = random.randint(0, w1 - self.fineSize)##
                    rh = random.randint(0, h1 - self.fineSize)##
                    B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    Bmask_O = Bmask_O.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    A_img = self.transform(B_img)
                    B_img = self.transform(Bmask_O)

               else:
                    path_original_mask = "./datasets/half/202/mask2_O.jpg"
                    Bmask_O =  Image.open(path_original_mask).convert('RGB')
                    path = "./datasets/half/202/train/2.jpg"
                    B_img = Image.open(path).convert('RGB')      
                    wB, hB = B_img.size

                    w1 = random.randint(wB-200, wB +200)
                    h1 = random.randint(hB-200, hB +200)
                    B_img=B_img.resize((w1,h1),Image.BILINEAR)
                    Bmask_O =Bmask_O.resize((w1,h1),Image.BILINEAR)

                    rw = random.randint(0, w1 - self.fineSize)
                    rh = random.randint(0, h1 - self.fineSize)
                    B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    Bmask_O = Bmask_O.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))
                    A_img = self.transform(B_img)
                    B_img = self.transform(Bmask_O)

          return {'A': A_img, 'B': B_img,
                'A_paths': path, 'B_paths': path,
                'A_start_point':[(rw, rh)]}

    def __len__(self):
        return self.size

    def name(self):
        return 'HalfDataset'
