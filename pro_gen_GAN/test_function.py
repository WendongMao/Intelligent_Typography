import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import copy
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np



def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    # if opt.isTrain and not opt.no_flip:
    #     transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



def test_func(opt_train, webpage, epoch='latest'):
    opt = copy.deepcopy(opt_train)
	#print(opt)
    # specify the directory to save the results during training
    opt.results_dir = './results/'
    opt.isTrain = False
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataroot = opt.dataroot + '/test'
    opt.model = 'test'
    opt.dataset_mode = 'single'
    opt.which_epoch = epoch
    opt.how_many = 50
    opt.phase = 'test'
	# opt.name = name

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
	# create website
	# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	# web_dir = os.path.join(opt.results_dir, opt.name)
	# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
	# test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
             break
        model.set_input(data,opt)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images_epoch(webpage, visuals, img_path, epoch)

    webpage.save()








       









