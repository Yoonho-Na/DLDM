import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from scripts.sample_diffusion import load_model

def denorm(x):
    x = x.float()
    x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    return x

def clamp(x):
    x = x.float()
    x = torch.clamp(x, -1., 1.)
    return x

def normalize_intensity(img_tensor, normalization="max"):
    """
    normalize non zero voxels only.
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        MAX, MIN = img_tensor.max(), img_tensor.min()
        if MAX != MIN:
            img_tensor = (img_tensor - MIN) / (MAX - MIN)
        else:
            pass
    return img_tensor

def DLDM_sample_images_npy(sample_dir, config_path, ckpt_path, type='png', glioma=True, n_samples=10, 
                   ddim_steps=200, ddim_eta=1., **kwargs):

    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    use_ddim = ddim_steps is not None
    with torch.no_grad():
        with model.ema_scope("Plotting"):
            i = 0
            while i < n_samples:
                if glioma:
                    # Glioma
                    c_0 = model.get_single_conditioning(0) # T2 
                    c_1 = model.get_single_conditioning(1) # FLAIR
                    c_2 = model.get_single_conditioning(2) # WB
                    c_3 = model.get_single_conditioning(3) # BB
                    c_4 = model.get_single_conditioning(4) # MASK
                    c_s = torch.cat([c_0, c_1, c_2, c_3, c_4], dim=0) # 5, 1, 128  <- class_label embeded vector
                else:
                    # Metastasis
                    c_0 = model.get_single_conditioning(5) # T2 
                    c_1 = model.get_single_conditioning(6) # FLAIR
                    c_2 = model.get_single_conditioning(7) # WB
                    c_3 = model.get_single_conditioning(8) # BB
                    c_4 = model.get_single_conditioning(9) # MASK
                    c_s = torch.cat([c_0, c_1, c_2, c_3, c_4], dim=0) # 5, 1, 128  <- class_label embeded vector

                samples_s, _ = model.sample_log(cond=c_s,batch_size=5,ddim=use_ddim, # 5, 4, 64, 64
                                                ddim_steps=ddim_steps,eta=ddim_eta)
                
                samples_s[1, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one
                samples_s[2, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one
                samples_s[3, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one
                samples_s[4, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one

                # samples_s[0].unsqueeze(0) = 1, 4, 64, 64 -> sample_0 = 1, 1, 256, 256

                sample_4 = model.decode_first_stage(samples_s[4].unsqueeze(0), force_not_quantize=False)
                sample_4 = model.mask_decode_first_stage(sample_4)
                sample_4 = multi_mask_vis((sample_4))
                unique = np.unique(sample_4)
                if len(np.unique(sample_4))==1: # if only background resample
                    continue
                if glioma: # glioma
                    print(unique)
                    if 0.25 in unique:
                        continue
                    sample_4 = np.where(sample_4<=0.25, 0., sample_4)
                    if np.count_nonzero(sample_4==1.) < 15:
                        continue
                else: # metastasis
                    if 0.5 in unique or 0.75 in unique or 1. in unique:
                        continue
                    sample_4 = np.where(sample_4>0.25, 0.25, sample_4)
                    if np.count_nonzero(sample_4) < 25:
                        continue

                sample_0 = model.decode_first_stage(samples_s[0].unsqueeze(0), force_not_quantize=False) # 1, 256, 256
                sample_1 = model.decode_first_stage(samples_s[1].unsqueeze(0), force_not_quantize=False)
                sample_2 = model.decode_first_stage(samples_s[2].unsqueeze(0), force_not_quantize=False)
                sample_3 = model.decode_first_stage(samples_s[3].unsqueeze(0), force_not_quantize=False)
                
                sample_0 = denorm(clamp(sample_0))
                sample_1 = denorm(clamp(sample_1))
                sample_2 = denorm(clamp(sample_2))
                sample_3 = denorm(clamp(sample_3))

                sample_0 = np.expand_dims(normalize_intensity(sample_0).cpu().numpy()[0][0], 2)
                sample_1 = np.expand_dims(normalize_intensity(sample_1).cpu().numpy()[0][0], 2)
                sample_2 = np.expand_dims(normalize_intensity(sample_2).cpu().numpy()[0][0], 2)
                sample_3 = np.expand_dims(normalize_intensity(sample_3).cpu().numpy()[0][0], 2)
                sample_4 = np.expand_dims(sample_4[0][0], 2)

                # PNG
                if type == 'png':
                    cv2.imwrite(sample_dir+'/images/T2_{}.png'.format(i), sample_0*255)
                    cv2.imwrite(sample_dir+'/images/FLAIR_{}.png'.format(i), sample_1*255)
                    cv2.imwrite(sample_dir+'/images/WB_{}.png'.format(i), sample_2*255)
                    cv2.imwrite(sample_dir+'/images/BB_{}.png'.format(i), sample_3*255)
                    cv2.imwrite(sample_dir+'/images/MASK_{}.png'.format(i), sample_4*255)      

                # npy
                elif type == 'npy':

                    np.save(sample_dir+'/npy/T2/T2_{}.npy'.format(i), sample_0[:, :, 0])
                    np.save(sample_dir+'/npy/FLAIR/FLAIR_{}.npy'.format(i), sample_1[:, :, 0])
                    np.save(sample_dir+'/npy/WB/WB_{}.npy'.format(i), sample_2[:, :, 0])
                    np.save(sample_dir+'/npy/BB/BB_{}.npy'.format(i), sample_3[:, :, 0])
                    np.save(sample_dir+'/npy/MASK/MASK_{}.npy'.format(i), sample_4[:, :, 0])

                else:
                    print('please configure file format')
                    quit()

                print('    [',i,'] sample saved')
                i += 1

def configure(config_path, ckpt_path, sample_dir, type, if_glioma, sample_number):
    config_path = config_path
    ckpt_path = ckpt_path
    print('Start on', os.path.splitext(ckpt_path)[0].split('/')[-1])
    sample_dir = sample_dir
    print('  Directory:', sample_dir)
    for seq in ['T2', 'FLAIR', 'WB', 'BB', 'MASK']:
        os.makedirs(os.path.join(sample_dir, type, seq), exist_ok=True)
    DLDM_sample_images_npy(sample_dir, config_path, ckpt_path, type=type, glioma=if_glioma, n_samples=sample_number, ddim_steps=50, ddim_eta=1.)
    print(os.path.splitext(ckpt_path)[1], 'Finished\n')

def multi_mask_vis(x): # For visualizing 2d matrix from range -1~1 to 0~1
    x=x.cpu().numpy()
    x = np.where(x > 0.75, 1., x) # necro
    x = np.where((x > 0.25) & (x < 0.75), 0.75, x) # enhance
    x = np.where((x > -0.25) & (x < 0.25), 0.5, x) # edema
    x = np.where((x > -0.75) & (x < -0.25), 0.25, x) # meta
    x = np.where(x < -0.75, 0., x) # background
    # x = torch.from_numpy(x)
    return x

def multi_mask_vis_(x): # For visualizing 2d matrix from range 0~1 to 0~1 (thresholded)
    mask=x
    mask = np.where(mask > 0.875, 1., mask) # necro
    mask = np.where((mask > 0.625) & (mask < 0.875), 0.75, mask) # enhance
    mask = np.where((mask > 0.375) & (mask < 0.625), 0.5, mask) # edema
    mask = np.where((mask > 0.125) & (mask < 0.375), 0.25, mask) # meta
    mask = np.where(mask < 0.125, 0., mask) # background
    x = mask
    return x

if __name__ == '__main__':

    config_path = 'configs/latent-diffusion/DLDM-kl-64x64x7.yaml'
    ckpt_path = "pretrained_DLDM/DLDM_pretrained/DLDM_epoch=000150.ckpt"

    metastasis_sample_dir = 'DLDM_samples/Metastasis'
    configure(config_path, ckpt_path, metastasis_sample_dir, type='npy', if_glioma=False, sample_number=5000)

    glioma_sample_dir = 'DLDM_samples/Glioma'
    configure(config_path, ckpt_path, glioma_sample_dir, type='npy', if_glioma=True, sample_number=5000)
