# Disentangled Latent Diffusion Model (DLDM)
This repo contains Pytorch implemented model definitions, pre-trained weights and training/sampling code for DLDMs.

### Installation guide
1. git clone this repo
2. Install the pkgs and activate envorionment
```
$ conda env create -f environment.yaml
$ conda activate dldm
```
### Download pre-trained model
We provide pretrained weights.
```
$ python scripts/pretrained_dldm.py
```
### Custom dataset
1. put your files (.jpg, .npy, .png, ...) in a folder `custom_folder`
2. create 2 text files a `xx_train.txt` and `xx_valid.txt` that point to the files in your training and test set respectively<br/>
`find $(pwd)/custom_folder/train -name "*.npy" > xx_train.txt`<br/>
`find $(pwd)/custom_folder/valid -name "*.npy" > xx_valid.txt`
```
${pwd}/custom_folder/train/
├── class1
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── class2
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── ...

${pwd}/custom_folder/valid/
├── class1
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── class2
│   ├── filename1.npy
│   ├── filename2.npy
│   ├── ...
├── ...
```
3. adapt `configs/custom_DAE.yaml` to point to these 2 files
4. run `python main.py --base configs/custom_DAE.yaml -t True --gpus 0,1` to
   train on two GPUs. Use `--gpus 0,` (with a trailing comma) to train on a single GPU.
   
## Summary

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
