#!/usr/bin/env bash
python test.py --dataroot ./datasets/fake --name 6_half_30x30 --model test --which_epoch latest --which_model_netG resnet_2x_6blocks --which_direction AtoB --dataset_mode single --norm batch --resize_or_crop none --fineSize 256 --gpu_ids 0