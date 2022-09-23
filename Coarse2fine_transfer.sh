python Forward_pro_gen.py 
python ./Structure_Net/style.py transfer --model-path "./Structure_Net/models/honghua.model" --source "./Gp1_prototype.jpg" --output "./Ns_result.jpg" 
python ./Texture_Net/texture_refine.py