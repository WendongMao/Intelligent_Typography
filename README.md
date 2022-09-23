# Artistic Text Style Transfer for Complex Texture and Structure

<table border="0" width='100%'>
 <tr align="center">	
  <td width="18.6%"><img src="https://github.com/WendongMao/Intelligent_Typography/picture/example.png" width="99%" ></td>
 </tr>
 <tr align="center">
  <td>source</td><td>adjustable stylistic degree of glyph</td><td>stylized text</td><td>application</td>
</tr>					 
 </table>
 <table border="0" width='100%'>
 <tr align="center">
  <td width="25%"><img src= "https://github.com/WendongMao/Intelligent_Typography/picture/love_higres.png" alt="" width="99%" ></td>	
  <td width="40%"><img src= "https://github.com/WendongMao/Intelligent_Typography/picture/spring_higres.png"alt="" width="99%" ></td>	
  <td width="35%"><img src="https://github.com/WendongMao/Intelligent_Typography/picture/tasty_higres.png" alt="" width="99%" ></td>			
 </tr>					 
 <tr align="center">
  <td>liquid artistic text rendering</td><td>smoke artistic text rendering</td>
</tr>	
</table>

This is a pytorch implementation of the paper.

Wendong Mao, Shuai Yang; Huihong Shi, Jiaying Liu, and Zhongfeng Wang,  "Intelligent Typography: Artistic Text Style Transfer for Complex Texture and Structure"  in IEEE Transactions on Multimedia: Regular Paper, 2022.

[[Paper]](XXXxxxxxx) | More about artistic text style transfer 

Please consider citing our paper if you find the software useful for your work.


## Usage: 

#### Prerequisites
- Python 3.6
- Pytorch 1.8.0
- matplotlib
- opencv
- scipy
- Pillow
- 

#### Install
- Clone this repo:
```
git clone https://github.com/WendongMao/Intelligent_Typography.git
cd ShapeMatchingGAN/src
```
## Testing Example

- Download pre-trained G_p and N_s models and input images from  [[Baidu Cloud]](https://xxxx)(code:ripi)
  -Save path of the  pre-trained G_p and N_s: `./pro_gen_GAN/checkpoints/Gp1/`,`./pro_gen_GAN/checkpoints/Gp2/`, `./Structure_Net/models/`,
#and it contains text styles "leaf"," flower","pitaya","wave", "ink", " peachblossom","branch"
- Artisic text style transfer using default style scale 0.0
  - Results for Gp, Ns and Nt can be found in `./Gp1_prototype.jpg`,`./Gp2_segmask.jpg`, `./Ns_result.jpg`, `./Nt_results.jpg`


```
python Forward_pro_gen.py \
python ./Structure_Net/style.py transfer --model-path "./Structure_Net/models/leaf.model" --source "./Gp1_prototype.jpg" --output "./Ns_result.jpg" \
python ./Texture_Net/texture_refine.py
```
- Artisic text style transfer with specified parameters
  - setting parameters --deforml of Forward_pro_gen.py from 1 to 3,5,7 means testing with multiple scales 
  - specify the input text name, output image path and name with text_name, result_dir and name, respectively


or just modifying and running

- For Artisic text style transfer with default parameters
```
Coarse2fine_transfer.sh
```

- For Artisic text style transfer with optional style-scales
```
sh ./Coarse2fine_deform.sh
```

- For relatively simple text styles with alternative inference steps
```
Coarse2fine_altern.sh
```


## Training Examples

### Training prototype generation pro-gen GAN
```
cd ./pro_gen_GAN
```
- Download style images from  [[Baidu Cloud]](https://xxxxx)(code:rjpi) to `./pro_gen_GAN/datasets/half/`
- Train G_p1 with default parameters

just modifying and running
```
sh ../script/train_Gp1.sh
```
Saved model can be found at `./checkpoints/Gp1`


- Train G_p2 with default parameters
just modifying and running
```
sh ../script/train_Gp2.sh
```
Saved model can be found at `./checkpoints/Gp2`



### Training Structure Refinement N_S
```
cd ./Structure_Net
```

- Train N_S with default parameters

 just modifying and running
```
sh ../script/launch_ShapeMGAN_structure.sh
```
Saved model can be found at `./models`


### Texture Refinement N_t

- N_t uses a pretrained VGG and does not require training
  - Download pre-trained VGG model from  [[Baidu Cloud]](https://xxx)(code:rjpi) 


### More

Three training examples are in the IPythonNotebook ShapeMatchingGAN.ipynb

Have fun :-)

### Try with your own style images
```
cd ./image_preparation
```
- Style image preparation for network training
  - Put the style image and text mask into the corresponding folders: `./style_img/` and `./text_mask/`.
  ```
  python make_restdir.py
  ```
  - A folder containing global/local images/masks can be generated (`./test_case/202xxx/`), and the degree of mask smoothness can be adjusted by changing the kernel size of gaussian filter.
  - Remove the generated folder(`202xxx/`) into the corresponding datapath `./pro_gen_GAN/datasets/half/`.
  - Noting: For the colour distinct style image, the code uses a threshold to judge the pixel of style image, then obtaining the 
binary masks. If the style image has complex color distribution, the generated binary mask will be mixed with the background. For these images with complex color, you can extract their binary masks by yourselves, and then replace the generated masks in the  target folder for network training.



### Contact

Wendong Mao

wdmao@smail.nju.edu.cn