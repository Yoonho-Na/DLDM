# DLDM
This repo contains Pytorch implemented model definitions, pre-trained weights and training/sampling code for disentangled latent diffusion models.

### Pretrained model
We provide pretrained weights.

### Custom dataset
1. install the repo with `conda env create -f environment.yaml`, `conda activate taming` and `pip install -e .`
1. put your .jpg, .npy, .png, ... files in a folder `custom_folder`
2. create 2 text files a `xx_train.txt` and `xx_valid.txt` that point to the files in your training and test set respectively<br/>
`find $(pwd)/custom_folder/train -name "*.npy" > xx_train.txt`<br/>
`find $(pwd)/custom_folder/valid -name "*.npy" > xx_valid.txt`
```
${pwd}/custom_folder/train/
├── T1
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── T2
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── ...

${pwd}/custom_folder/valid/
├── T1
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── T2
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── ...
```
3. adapt `configs/custom_DAE.yaml` to point to these 2 files
4. run `python main.py --base configs/custom_DAE.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.
   
## Disentangled Latent Diffusion Model (DLDM)

### DLDM component
* Disentangled AutoEncoder (DAE)
* Denoising Network
* Embedding Network

## Training process
<p align="center">
 <img width="900" src="https://github.com/Yoonho-Na/DLDM/blob/main/figures/main_figure_bg.png?raw=true">
</p>

## Multi-modal image generation
<p align="center">
 <img width="900" src="https://github.com/Yoonho-Na/DLDM/blob/main/figures/multi-modal_generation_bg.png?raw=true">
</p>

## Generation of brain metastasis MRIs
<p align="center">
 <img width="900" src="https://github.com/Yoonho-Na/DLDM/blob/main/figures/original%20vs%20DLDM%20vs%20StyleGAN2_bg.png?raw=true">
</p>

## Disentanglement study
### Mixture of structure and style latents
<p align="center">
 <img width="900" src="https://github.com/Yoonho-Na/DLDM/blob/main/figures/mix_structure_style_bg.png?raw=true">
</p>
### Walking on the style latents
<p align="center">
 <img width="900" src="https://github.com/Yoonho-Na/DLDM/blob/main/figures/style_interpolation_bg.png?raw=true">
</p>
