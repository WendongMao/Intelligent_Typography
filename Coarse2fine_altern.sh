python Forward_pro_gen.py --Gp1path "./pro_gen_GAN/checkpoints/Gp1/latest_net_G.pth"  --picpath "./pro_gen_GAN/image_preparation/test_case/202pitaya/train/2.jpg" --tpath "./image_preparation/text_mask/han1.jpg" --Gp2path "no_path" 
python ./Texture_Net/texture_refine.py --picpath "./pro_gen_GAN/image_preparation/test_case/202pitaya/train/1.jpg" --Nsopath "./Gp1_prototype.jpg" --Gp2opath "no_path"
