# Disentangled Latent Diffusion Model (DLDM)
This repo contains Pytorch implemented model definitions, pre-trained weights and training/sampling code for DLDMs.

### Installation guide
1. git clone this repo
2. Install the pkgs and activate envorionment
```
$ git clone git@github.com:Yoonho-Na/DLDM.git
$ cd DLDM
$ conda env create -f environment.yaml
$ conda activate dldm
```
### Download pre-trained model
We provide pretrained weights.
```
$ python scripts/pretrained_dldm.py
```
### Sampling
```
$ python sample.py
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

### Generated dataset
- https://drive.google.com/drive/folders/1o2G57UrSs3v4whlVAVDVAGO-N3iRraEk?usp=sharing
