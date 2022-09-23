import numpy as np
import torch
import os
import argparse
import time
import torch.nn as nn

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from network import ImageTransformNet
from vgg import Vgg16
# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 5e10  
#STYLE_WEIGHT = 5e8
#STYLE_WEIGHT = 1000
TV_WEIGHT = 1e-6
EDGE_WEIGHT = 5e10

def train(args):          
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        #torch.cuda.set_device(args.gpu)
        #print ("Current device: %d" %torch.cuda.current_device())

    dtype = torch.cuda.FloatTensor
    # visualization of training controlled by flag
    visualize = (args.visualize != None)
    if (visualize):
        img_transform_512 = transforms.Compose([
            transforms.Scale(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        testImage_amber = utils.load_image("../Gp1_prototype.jpg")
        testImage_amber = img_transform_512(testImage_amber)
        testImage_amber = Variable(testImage_amber.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # define network
    image_transformer = ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().type(dtype)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style)
    style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).type(dtype)
    style_name = os.path.split(args.style_image)[-1].split('.')[0]

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]
    

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, (x, label) in enumerate(train_loader):
            
            img_batch_read = len(x)
            img_count += img_batch_read
            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).type(dtype)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()
            
            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss 

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item()
                            )
                print(status)

            if ((batch_num + 1) % 1000 == 0) and (visualize):
                image_transformer.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                if not os.path.exists("visualization/%s" %style_name):
                    os.makedirs("visualization/%s" %style_name)

                outputTestImage_amber = image_transformer(testImage_amber).cpu()
                amber_path = "visualization/%s/G1_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(amber_path, outputTestImage_amber.data[0])
                
                print("images saved")
                image_transformer.train()

    print("training finishing!")
    # save model
    image_transformer.eval()

    if use_cuda:
        image_transformer.cpu()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    torch.save(image_transformer.state_dict(), filename)
    
    if use_cuda:
        image_transformer.cuda()

def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        #print ("Current device: %d" %torch.cuda.current_device())

    # content image
    img_transform = transforms.Compose([
            #transforms.Scale(256),                  # scale shortest side to image_size
            #transforms.CenterCrop(256),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform(content)
    content = content.unsqueeze(0)
    content = Variable(content.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype).eval()
    style_model.cuda()
    #style_model = nn.DataParallel(style_model)

    state_dict =torch.load(args.model_path)

    style_model.load_state_dict(state_dict)


    # process input image
    stylized = style_model(content).cpu()
    utils.save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, required=True, help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True, default="/models/udnie.model", help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, default="/content_imgs/dan.jpg", required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, default="/result/1.png", required=True, help="file name for stylized output image")
    style_parser.add_argument("--gpu", type=int, default=0, help="ID of GPU to be used")

    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        print ("Training Structure Network!")
        train(args)
    elif (args.subcommand == "transfer"):
        print ("Forward Structure Refinement!\n")
        style_transfer(args)
    else:
        print("invalid command")

if __name__ == '__main__':
    main()








