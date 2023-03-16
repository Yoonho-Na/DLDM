import torch
import numpy as np

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
import os, glob, shutil, copy
from PIL import Image
import random

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

def interpolate_tensor(tensor_a, tensor_b, alpha):
    # Define two tensors to interpolate between
    tensor_a = tensor_a
    tensor_b = tensor_b

    # Define the interpolation factor
    alpha = alpha

    # Interpolate between the two tensors
    interpolated_tensor = alpha * tensor_a + (1 - alpha) * tensor_b

    return interpolated_tensor

def interpolate_styles(sample_dir, config_path, ckpt_path, n_samples=10, 
                   ddim_steps=200, ddim_eta=1., **kwargs):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    # original_list= glob.glob('/home/yhn/Meta_synthesis/data/Severance_multi_without_label/train'+'/*.npy')

    use_ddim = ddim_steps is not None
    with torch.no_grad():
        with model.ema_scope("Plotting"):
            for i in range(n_samples):

                c_0 = model.get_single_conditioning(0) # T1
                c_1 = model.get_single_conditioning(1) # T2
                c_2 = model.get_single_conditioning(2) # FLAIR
                c_3 = model.get_single_conditioning(3) # WB
                c_4 = model.get_single_conditioning(4) # BB
                c_s = torch.cat([c_0, c_1, c_2, c_3, c_4], dim=0) # 5, 1, 128  <- class_label embeded vector

                z1, _ = model.sample_log(cond=c_s,batch_size=5,ddim=use_ddim, # 5, 4, 64, 64
                                                ddim_steps=ddim_steps,eta=ddim_eta)
                
                # fix all structure to 1st data
                for b in range(z1.shape[0]):
                    z1[b, :2, :, :] = z1[0, :2, :, : ]

                structure = z1[0, :2, :, : ].unsqueeze(0)
                t1_style = z1[0, 2:, :, :].unsqueeze(0)
                t2_style = z1[1, 2:, :, :].unsqueeze(0)

                inter_style_0 = interpolate_tensor(t1_style, t2_style, 0.)
                inter_style_1 = interpolate_tensor(t1_style, t2_style, 0.1)
                inter_style_2 = interpolate_tensor(t1_style, t2_style, 0.2)
                inter_style_3 = interpolate_tensor(t1_style, t2_style, 0.3)
                inter_style_4 = interpolate_tensor(t1_style, t2_style, 0.4)
                inter_style_5 = interpolate_tensor(t1_style, t2_style, 0.5)
                inter_style_6 = interpolate_tensor(t1_style, t2_style, 0.6)
                inter_style_7 = interpolate_tensor(t1_style, t2_style, 0.7)
                inter_style_8 = interpolate_tensor(t1_style, t2_style, 0.8)
                inter_style_9 = interpolate_tensor(t1_style, t2_style, 0.9)
                inter_style_10 = interpolate_tensor(t1_style, t2_style, 1.)

                iterpolation_0 = torch.cat([structure, inter_style_0], dim=1)
                iterpolation_1 = torch.cat([structure, inter_style_1], dim=1)
                iterpolation_2 = torch.cat([structure, inter_style_2], dim=1)
                iterpolation_3 = torch.cat([structure, inter_style_3], dim=1)
                iterpolation_4 = torch.cat([structure, inter_style_4], dim=1)
                iterpolation_5 = torch.cat([structure, inter_style_5], dim=1)
                iterpolation_6 = torch.cat([structure, inter_style_6], dim=1)
                iterpolation_7 = torch.cat([structure, inter_style_7], dim=1)
                iterpolation_8 = torch.cat([structure, inter_style_8], dim=1)
                iterpolation_9 = torch.cat([structure, inter_style_9], dim=1)
                iterpolation_10 = torch.cat([structure, inter_style_10], dim=1) # 1, 4, 64, 64

                inter_0 = model.decode_first_stage(iterpolation_0)[0][0]
                inter_1 = model.decode_first_stage(iterpolation_1)[0][0]
                inter_2 = model.decode_first_stage(iterpolation_2)[0][0]
                inter_3 = model.decode_first_stage(iterpolation_3)[0][0]
                inter_4 = model.decode_first_stage(iterpolation_4)[0][0]
                inter_5 = model.decode_first_stage(iterpolation_5)[0][0]
                inter_6 = model.decode_first_stage(iterpolation_6)[0][0]
                inter_7 = model.decode_first_stage(iterpolation_7)[0][0]
                inter_8 = model.decode_first_stage(iterpolation_8)[0][0]
                inter_9 = model.decode_first_stage(iterpolation_9)[0][0]
                inter_10 = model.decode_first_stage(iterpolation_10)[0][0] #256, 256
                all_inter = torch.cat([normalize_intensity(inter_0), 
                                       normalize_intensity(inter_1), 
                                       normalize_intensity(inter_2), 
                                       normalize_intensity(inter_3), 
                                       normalize_intensity(inter_4), 
                                       normalize_intensity(inter_5), 
                                       normalize_intensity(inter_6), 
                                       normalize_intensity(inter_7), 
                                       normalize_intensity(inter_8), 
                                       normalize_intensity(inter_9), 
                                       normalize_intensity(inter_10)], dim=1)
                img = Image.fromarray(np.uint8(all_inter.cpu().numpy()*255))
                img.save(sample_dir+'/images/T1_T2_interpolated_{}.png'.format(i))
                print('    [',i,'] sample saved')


def interpolate_structures(sample_dir, config_path, ckpt_path, n_samples=10, 
                   ddim_steps=200, ddim_eta=1., **kwargs):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    # original_list= glob.glob('/home/yhn/Meta_synthesis/data/Severance_multi_without_label/train'+'/*.npy')

    use_ddim = ddim_steps is not None
    with torch.no_grad():
        with model.ema_scope("Plotting"):
            for i in range(n_samples):

                c_0 = model.get_single_conditioning(0) # T1
                c_1 = model.get_single_conditioning(1) # T2
                c_2 = model.get_single_conditioning(2) # FLAIR
                c_3 = model.get_single_conditioning(3) # WB
                c_4 = model.get_single_conditioning(4) # BB
                c_s = torch.cat([c_0, c_1, c_2, c_3, c_4], dim=0) # 5, 1, 128  <- class_label embeded vector

                z1, _ = model.sample_log(cond=c_s,batch_size=5,ddim=use_ddim, # 5, 4, 64, 64
                                                ddim_steps=ddim_steps,eta=ddim_eta)
                z2, _ = model.sample_log(cond=c_s,batch_size=5,ddim=use_ddim, # 5, 4, 64, 64
                                ddim_steps=ddim_steps,eta=ddim_eta)
                
                # fix all structure to 1st data
                for b in range(z1.shape[0]):
                    z1[b, :2, :, :] = z1[0, :2, :, : ]
                for b in range(z2.shape[0]):
                    z2[b, :2, :, :] = z2[0, :2, :, : ]

                structure1 = z1[0, :2, :, :].unsqueeze(0)
                structure2 = z2[0, :2, :, :].unsqueeze(0)
                t1_style = z1[0, 2:, :, :].unsqueeze(0)
                t2_style = z1[1, 2:, :, :].unsqueeze(0)

                inter_structure_0 = interpolate_tensor(structure1, structure2, 0.)
                inter_structure_1 = interpolate_tensor(structure1, structure2, 0.1)
                inter_structure_2 = interpolate_tensor(structure1, structure2, 0.2)
                inter_structure_3 = interpolate_tensor(structure1, structure2, 0.3)
                inter_structure_4 = interpolate_tensor(structure1, structure2, 0.4)
                inter_structure_5 = interpolate_tensor(structure1, structure2, 0.5)
                inter_structure_6 = interpolate_tensor(structure1, structure2, 0.6)
                inter_structure_7 = interpolate_tensor(structure1, structure2, 0.7)
                inter_structure_8 = interpolate_tensor(structure1, structure2, 0.8)
                inter_structure_9 = interpolate_tensor(structure1, structure2, 0.9)
                inter_structure_10 = interpolate_tensor(structure1, structure2, 1.)

                iterpolation_0 = torch.cat([inter_structure_0, t1_style], dim=1)
                iterpolation_1 = torch.cat([inter_structure_1, t1_style], dim=1)
                iterpolation_2 = torch.cat([inter_structure_2, t1_style], dim=1)
                iterpolation_3 = torch.cat([inter_structure_3, t1_style], dim=1)
                iterpolation_4 = torch.cat([inter_structure_4, t1_style], dim=1)
                iterpolation_5 = torch.cat([inter_structure_5, t1_style], dim=1)
                iterpolation_6 = torch.cat([inter_structure_6, t1_style], dim=1)
                iterpolation_7 = torch.cat([inter_structure_7, t1_style], dim=1)
                iterpolation_8 = torch.cat([inter_structure_8, t1_style], dim=1)
                iterpolation_9 = torch.cat([inter_structure_9, t1_style], dim=1)
                iterpolation_10 = torch.cat([inter_structure_10, t1_style], dim=1) # 1, 4, 64, 64

                inter_0 = model.decode_first_stage(iterpolation_0)[0][0]
                inter_1 = model.decode_first_stage(iterpolation_1)[0][0]
                inter_2 = model.decode_first_stage(iterpolation_2)[0][0]
                inter_3 = model.decode_first_stage(iterpolation_3)[0][0]
                inter_4 = model.decode_first_stage(iterpolation_4)[0][0]
                inter_5 = model.decode_first_stage(iterpolation_5)[0][0]
                inter_6 = model.decode_first_stage(iterpolation_6)[0][0]
                inter_7 = model.decode_first_stage(iterpolation_7)[0][0]
                inter_8 = model.decode_first_stage(iterpolation_8)[0][0]
                inter_9 = model.decode_first_stage(iterpolation_9)[0][0]
                inter_10 = model.decode_first_stage(iterpolation_10)[0][0] #256, 256
                all_inter = torch.cat([normalize_intensity(inter_0), 
                                       normalize_intensity(inter_1), 
                                       normalize_intensity(inter_2), 
                                       normalize_intensity(inter_3), 
                                       normalize_intensity(inter_4), 
                                       normalize_intensity(inter_5), 
                                       normalize_intensity(inter_6), 
                                       normalize_intensity(inter_7), 
                                       normalize_intensity(inter_8), 
                                       normalize_intensity(inter_9), 
                                       normalize_intensity(inter_10)], dim=1)
                img = Image.fromarray(np.uint8(all_inter.cpu().numpy()*255))
                img.save(sample_dir+'/images/T1_structure_interpolated_{}.png'.format(i))
                print('    [',i,'] sample saved')




def sample_images_npy(sample_dir, config_path, ckpt_path, n_samples=10, 
                   ddim_steps=200, ddim_eta=1., structure=True, style=True, experiment=False, **kwargs):

    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    original_list= glob.glob('/home/yhn/Meta_synthesis/data/Severance_multi_without_label/train'+'/*.npy')


    use_ddim = ddim_steps is not None
    with torch.no_grad():
        with model.ema_scope("Plotting"):
            for i in range(n_samples):

                c_0 = model.get_single_conditioning(0) # T1
                c_1 = model.get_single_conditioning(1) # T2
                c_2 = model.get_single_conditioning(2) # FLAIR
                c_3 = model.get_single_conditioning(3) # WB
                c_4 = model.get_single_conditioning(4) # BB
                c_s = torch.cat([c_0, c_1, c_2, c_3, c_4], dim=0) # 5, 1, 128  <- class_label embeded vector

                samples_s, _ = model.sample_log(cond=c_s,batch_size=5,ddim=use_ddim, # 5, 4, 64, 64
                                                ddim_steps=ddim_steps,eta=ddim_eta)

                if experiment == False:
                    samples_s[1, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one
                    samples_s[2, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one
                    samples_s[3, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one
                    samples_s[4, :2, :, :] = samples_s[0, :2, :, :] # fix first channel to this one

                    # samples_s[0].unsqueeze(0) = 1, 4, 64, 64 -> sample_0 = 1, 1, 256, 256
                    sample_0 = model.decode_first_stage(samples_s[0].unsqueeze(0)) # 1, 256, 256
                    sample_1 = model.decode_first_stage(samples_s[1].unsqueeze(0))
                    sample_2 = model.decode_first_stage(samples_s[2].unsqueeze(0))
                    sample_3 = model.decode_first_stage(samples_s[3].unsqueeze(0))
                    sample_4 = model.decode_first_stage(samples_s[4].unsqueeze(0))

                    # sample_0 = denorm(clamp(sample_0))
                    # sample_1 = denorm(clamp(sample_1))
                    # sample_2 = denorm(clamp(sample_2))
                    # sample_3 = denorm(clamp(sample_3))
                    # sample_4 = denorm(clamp(sample_4))

                    sample_0 = denorm(sample_0)
                    sample_1 = denorm(sample_1)
                    sample_2 = denorm(sample_2)
                    sample_3 = denorm(sample_3)
                    sample_4 = denorm(sample_4)

                    # structure
                    if structure == True:
                        np.save(sample_dir+'/structure/{}'.format(i), samples_s[0, :2, :, :].cpu().numpy())

                    # style
                    if style == True:
                        np.save(sample_dir+'/style/{}_0'.format(i), samples_s[0, 2:, :, :].cpu().numpy()) 
                        np.save(sample_dir+'/style/{}_1'.format(i), samples_s[1, 2:, :, :].cpu().numpy())
                        np.save(sample_dir+'/style/{}_2'.format(i), samples_s[2, 2:, :, :].cpu().numpy())
                        np.save(sample_dir+'/style/{}_3'.format(i), samples_s[3, 2:, :, :].cpu().numpy())
                        np.save(sample_dir+'/style/{}_4'.format(i), samples_s[4, 2:, :, :].cpu().numpy())

                    # images
                    np.save(sample_dir+'/images/T1_{}'.format(i), normalize_intensity(sample_0).cpu().numpy())
                    np.save(sample_dir+'/images/T2_{}'.format(i), normalize_intensity(sample_1).cpu().numpy())
                    np.save(sample_dir+'/images/FLAIR_{}'.format(i), normalize_intensity(sample_2).cpu().numpy())
                    np.save(sample_dir+'/images/WB_{}'.format(i), normalize_intensity(sample_3).cpu().numpy())
                    np.save(sample_dir+'/images/BB_{}'.format(i), normalize_intensity(sample_4).cpu().numpy())
                    print('    [',i,'] sample saved')

                if experiment == True:
                    # ######################################################################################
                    
                    original_file = random.sample(original_list, 1)
                    x = np.load(original_file[0])
                    x = torch.from_numpy((x/0.5 - 1.0).astype(np.float32)).unsqueeze(0).to("cuda") # 1, 5, 256, 256
                    original_encoder_posterior = model.encode_first_stage(x)
                    original_z = model.get_first_stage_encoding(original_encoder_posterior).detach() # 1, 12, 64, 64
                    original_structure_style0 = torch.cat([original_z[:, 0:2, :, :], original_z[:, 2:4, :, :]], dim=1)
                    original_structure_style1 = torch.cat([original_z[:, 0:2, :, :], original_z[:, 4:6, :, :]], dim=1)
                    original_structure_style2 = torch.cat([original_z[:, 0:2, :, :], original_z[:, 6:8, :, :]], dim=1)
                    original_structure_style3 = torch.cat([original_z[:, 0:2, :, :], original_z[:, 8:10, :, :]], dim=1)
                    original_structure_style4 = torch.cat([original_z[:, 0:2, :, :], original_z[:, 10:12, :, :]], dim=1)
                    original_z=torch.cat([original_structure_style0, original_structure_style1, original_structure_style2, original_structure_style3, original_structure_style4], dim=0)
                    
                    original_sample = torch.cat([normalize_intensity(x[0][0]), 
                                        normalize_intensity(x[0][1]), 
                                        normalize_intensity(x[0][2]), 
                                        normalize_intensity(x[0][3]), 
                                        normalize_intensity(x[0][4])], dim=1) #256, 1280
                    original_sample = Image.fromarray(np.uint8(original_sample.cpu().numpy()*255))
                    original_sample.save(sample_dir+'/images/original_sample_{}.png'.format(i))

                    z1 = samples_s # 5, 4, 64, 64
                    z2, _ = model.sample_log(cond=c_s,batch_size=5,ddim=use_ddim, # 5, 4, 64, 64
                                                ddim_steps=ddim_steps,eta=ddim_eta)
                    # fix all structure to 1st data
                    for b in range(z1.shape[0]):
                        z1[b, :2, :, :] = z1[0, :2, :, : ]
                        z2[b, :2, :, :] = z2[0, :2, :, : ]

                    z1_structure_z2_style = z1.clone().detach()
                    z2_structure_z1_style = z2.clone().detach()
                    z1_structure_z2_style[:, 2:, :, :] = z2[:, 2:, :, :]
                    z2_structure_z1_style[:, 2:, :, :] = z1[:, 2:, :, :]

                    z1_structure_no_style = z1.clone().detach()
                    z2_structure_no_style = z2.clone().detach()
                    z1_structure_no_style[:, 2:, :, :] = 0
                    z2_structure_no_style[:, 2:, :, :] = 0

                    z1_style_no_structure = z1.clone().detach()
                    z2_style_no_structure = z2.clone().detach()
                    z1_style_no_structure[:, :2, :, :] = 0
                    z2_style_no_structure[:, :2, :, :] = 0
                    
                    original_z_structure_z1_style = original_z.clone().detach()
                    original_z_structure_z2_style = original_z.clone().detach()
                    original_z_structure_z1_style[:, 2:, :, :] = z1[:, 2:, :, :]
                    original_z_structure_z2_style[:, 2:, :, :] = z2[:, 2:, :, :]

                    original_z_structure = original_z.clone().detach()
                    original_z_style = original_z.clone().detach()
                    original_z_structure[:, 2:, :, :] = 0
                    original_z_style[:, :2, :, :] = 0

                    z1_structure_original_style = z1.clone().detach()
                    z2_structure_original_style = z2.clone().detach()
                    z1_structure_original_style[:, 2:, :, :] = original_z[:, 2:, :, :]
                    z2_structure_original_style[:, 2:, :, :] = original_z[:, 2:, :, :]

                    # show normal inferred image
                    z1_sample = model.decode_first_stage(z1) # 5, 1, 256, 256
                    z1_sample = torch.cat([normalize_intensity(z1_sample[0][0]), 
                                        normalize_intensity(z1_sample[1][0]), 
                                        normalize_intensity(z1_sample[2][0]), 
                                        normalize_intensity(z1_sample[3][0]), 
                                        normalize_intensity(z1_sample[4][0])], dim=1) #256, 1280
                    z1_sample = Image.fromarray(np.uint8(z1_sample.cpu().numpy()*255))
                    z1_sample.save(sample_dir+'/images/sample_z1_{}.png'.format(i))

                    z2_sample = model.decode_first_stage(z2) # 5, 1, 256, 256
                    z2_sample = torch.cat([normalize_intensity(z2_sample[0][0]), 
                                        normalize_intensity(z2_sample[1][0]), 
                                        normalize_intensity(z2_sample[2][0]), 
                                        normalize_intensity(z2_sample[3][0]), 
                                        normalize_intensity(z2_sample[4][0])], dim=1)
                    z2_sample = Image.fromarray(np.uint8(z2_sample.cpu().numpy()*255))
                    z2_sample.save(sample_dir+'/images/sample_z2_{}.png'.format(i))

                    # show structure and style switched image
                    z1_structure_z2_style_sample = model.decode_first_stage(z1_structure_z2_style) # 5, 1, 256, 256
                    z1_structure_z2_style_sample = torch.cat([normalize_intensity(z1_structure_z2_style_sample[0][0]), 
                                                            normalize_intensity(z1_structure_z2_style_sample[1][0]), 
                                                            normalize_intensity(z1_structure_z2_style_sample[2][0]), 
                                                            normalize_intensity(z1_structure_z2_style_sample[3][0]), 
                                                            normalize_intensity(z1_structure_z2_style_sample[4][0])], dim=1)
                    z1_structure_z2_style_sample = Image.fromarray(np.uint8(z1_structure_z2_style_sample.cpu().numpy()*255))
                    z1_structure_z2_style_sample.save(sample_dir+'/images/z1_structure_z2_style_{}.png'.format(i))

                    z2_structure_z1_style_sample = model.decode_first_stage(z2_structure_z1_style) # 5, 1, 256, 256
                    z2_structure_z1_style_sample = torch.cat([normalize_intensity(z2_structure_z1_style_sample[0][0]), 
                                                            normalize_intensity(z2_structure_z1_style_sample[1][0]), 
                                                            normalize_intensity(z2_structure_z1_style_sample[2][0]), 
                                                            normalize_intensity(z2_structure_z1_style_sample[3][0]), 
                                                            normalize_intensity(z2_structure_z1_style_sample[4][0])], dim=1)
                    z2_structure_z1_style_sample = Image.fromarray(np.uint8(z2_structure_z1_style_sample.cpu().numpy()*255))
                    z2_structure_z1_style_sample.save(sample_dir+'/images/z2_structure_z1_style_{}.png'.format(i))

                    # show no style
                    z1_structure_no_style_sample = model.decode_first_stage(z1_structure_no_style) # 5, 1, 256, 256
                    z1_structure_no_style_sample = torch.cat([normalize_intensity(z1_structure_no_style_sample[0][0]), 
                                                            normalize_intensity(z1_structure_no_style_sample[1][0]), 
                                                            normalize_intensity(z1_structure_no_style_sample[2][0]), 
                                                            normalize_intensity(z1_structure_no_style_sample[3][0]), 
                                                            normalize_intensity(z1_structure_no_style_sample[4][0])], dim=1)
                    z1_structure_no_style_sample = Image.fromarray(np.uint8(z1_structure_no_style_sample.cpu().numpy()*255))
                    z1_structure_no_style_sample.save(sample_dir+'/images/z1_structure_no_style_{}.png'.format(i))

                    z2_structure_no_style_sample = model.decode_first_stage(z2_structure_no_style) # 5, 1, 256, 256
                    z2_structure_no_style_sample = torch.cat([normalize_intensity(z2_structure_no_style_sample[0][0]), 
                                                            normalize_intensity(z2_structure_no_style_sample[1][0]), 
                                                            normalize_intensity(z2_structure_no_style_sample[2][0]), 
                                                            normalize_intensity(z2_structure_no_style_sample[3][0]), 
                                                            normalize_intensity(z2_structure_no_style_sample[4][0])], dim=1)
                    z2_structure_no_style_sample = Image.fromarray(np.uint8(z2_structure_no_style_sample.cpu().numpy()*255))
                    z2_structure_no_style_sample.save(sample_dir+'/images/z2_structure_no_style_{}.png'.format(i))

                    # show no structure
                    z1_style_no_structure_sample = model.decode_first_stage(z1_style_no_structure) # 5, 1, 256, 256
                    z1_style_no_structure_sample = torch.cat([normalize_intensity(z1_style_no_structure_sample[0][0]), 
                                                            normalize_intensity(z1_style_no_structure_sample[1][0]), 
                                                            normalize_intensity(z1_style_no_structure_sample[2][0]), 
                                                            normalize_intensity(z1_style_no_structure_sample[3][0]), 
                                                            normalize_intensity(z1_style_no_structure_sample[4][0])], dim=1)
                    z1_style_no_structure_sample = Image.fromarray(np.uint8(z1_style_no_structure_sample.cpu().numpy()*255))
                    z1_style_no_structure_sample.save(sample_dir+'/images/z1_style_no_structure_{}.png'.format(i))

                    z2_style_no_structure_sample = model.decode_first_stage(z2_style_no_structure) # 5, 1, 256, 256
                    z2_style_no_structure_sample = torch.cat([normalize_intensity(z2_style_no_structure_sample[0][0]), 
                                                            normalize_intensity(z2_style_no_structure_sample[1][0]), 
                                                            normalize_intensity(z2_style_no_structure_sample[2][0]), 
                                                            normalize_intensity(z2_style_no_structure_sample[3][0]), 
                                                            normalize_intensity(z2_style_no_structure_sample[4][0])], dim=1)
                    z2_style_no_structure_sample = Image.fromarray(np.uint8(z2_style_no_structure_sample.cpu().numpy()*255))
                    z2_style_no_structure_sample.save(sample_dir+'/images/z2_style_no_structure_{}.png'.format(i))

                    ####

                    original_z_structure_z1_style_sample = model.decode_first_stage(original_z_structure_z1_style) # 5, 1, 256, 256
                    original_z_structure_z1_style_sample = torch.cat([normalize_intensity(original_z_structure_z1_style_sample[0][0]), 
                                                            normalize_intensity(original_z_structure_z1_style_sample[1][0]), 
                                                            normalize_intensity(original_z_structure_z1_style_sample[2][0]), 
                                                            normalize_intensity(original_z_structure_z1_style_sample[3][0]), 
                                                            normalize_intensity(original_z_structure_z1_style_sample[4][0])], dim=1)
                    original_z_structure_z1_style_sample = Image.fromarray(np.uint8(original_z_structure_z1_style_sample.cpu().numpy()*255))
                    original_z_structure_z1_style_sample.save(sample_dir+'/images/original_z_structure_z1_style_{}.png'.format(i))

                    original_z_structure_z2_style_sample = model.decode_first_stage(original_z_structure_z2_style) # 5, 1, 256, 256
                    original_z_structure_z2_style_sample = torch.cat([normalize_intensity(original_z_structure_z2_style_sample[0][0]), 
                                                            normalize_intensity(original_z_structure_z2_style_sample[1][0]), 
                                                            normalize_intensity(original_z_structure_z2_style_sample[2][0]), 
                                                            normalize_intensity(original_z_structure_z2_style_sample[3][0]), 
                                                            normalize_intensity(original_z_structure_z2_style_sample[4][0])], dim=1)
                    original_z_structure_z2_style_sample = Image.fromarray(np.uint8(original_z_structure_z2_style_sample.cpu().numpy()*255))
                    original_z_structure_z2_style_sample.save(sample_dir+'/images/original_z_structure_z2_style_{}.png'.format(i))

                    ####

                    original_z_structure_sample = model.decode_first_stage(original_z_structure) # 5, 1, 256, 256
                    original_z_structure_sample = torch.cat([normalize_intensity(original_z_structure_sample[0][0]), 
                                                            normalize_intensity(original_z_structure_sample[1][0]), 
                                                            normalize_intensity(original_z_structure_sample[2][0]), 
                                                            normalize_intensity(original_z_structure_sample[3][0]), 
                                                            normalize_intensity(original_z_structure_sample[4][0])], dim=1)
                    original_z_structure_sample = Image.fromarray(np.uint8(original_z_structure_sample.cpu().numpy()*255))
                    original_z_structure_sample.save(sample_dir+'/images/original_z_structure{}.png'.format(i))

                    original_z_style_sample = model.decode_first_stage(original_z_style) # 5, 1, 256, 256
                    original_z_style_sample = torch.cat([normalize_intensity(original_z_style_sample[0][0]), 
                                                            normalize_intensity(original_z_style_sample[1][0]), 
                                                            normalize_intensity(original_z_style_sample[2][0]), 
                                                            normalize_intensity(original_z_style_sample[3][0]), 
                                                            normalize_intensity(original_z_style_sample[4][0])], dim=1)
                    original_z_style_sample = Image.fromarray(np.uint8(original_z_style_sample.cpu().numpy()*255))
                    original_z_style_sample.save(sample_dir+'/images/original_z_style{}.png'.format(i))

                    ####

                    z1_structure_original_style_sample = model.decode_first_stage(z1_structure_original_style) # 5, 1, 256, 256
                    z1_structure_original_style_sample = torch.cat([normalize_intensity(z1_structure_original_style_sample[0][0]), 
                                                            normalize_intensity(z1_structure_original_style_sample[1][0]), 
                                                            normalize_intensity(z1_structure_original_style_sample[2][0]), 
                                                            normalize_intensity(z1_structure_original_style_sample[3][0]), 
                                                            normalize_intensity(z1_structure_original_style_sample[4][0])], dim=1)
                    z1_structure_original_style_sample = Image.fromarray(np.uint8(z1_structure_original_style_sample.cpu().numpy()*255))
                    z1_structure_original_style_sample.save(sample_dir+'/images/z1_structure_original_style{}.png'.format(i))

                    z2_structure_original_style_sample = model.decode_first_stage(z2_structure_original_style) # 5, 1, 256, 256
                    z2_structure_original_style_sample = torch.cat([normalize_intensity(z2_structure_original_style_sample[0][0]), 
                                                            normalize_intensity(z2_structure_original_style_sample[1][0]), 
                                                            normalize_intensity(z2_structure_original_style_sample[2][0]), 
                                                            normalize_intensity(z2_structure_original_style_sample[3][0]), 
                                                            normalize_intensity(z2_structure_original_style_sample[4][0])], dim=1)
                    z2_structure_original_style_sample = Image.fromarray(np.uint8(z2_structure_original_style_sample.cpu().numpy()*255))
                    z2_structure_original_style_sample.save(sample_dir+'/images/z2_structure_original_style{}.png'.format(i))

                    print('    [',i,'] sample saved')


def configure(config_path, ckpt_path, sample_dir, sample_number, structure=True, style=True, experiment=False, interpolate_style=False, interpolate_structure=False):
    config_path = config_path
    ckpt_path = ckpt_path
    print('Start on', os.path.splitext(ckpt_path)[0].split('/')[-1])
    sample_dir = sample_dir
    print('  Directory:', sample_dir)
    os.makedirs(os.path.join(sample_dir, 'images'), exist_ok=True)
    if interpolate_style == True:
        interpolate_styles(sample_dir, config_path, ckpt_path, n_samples=sample_number)
    elif interpolate_structure == True:
        interpolate_structures(sample_dir, config_path, ckpt_path, n_samples=sample_number)
    else:
        if structure == True:
            os.makedirs(os.path.join(sample_dir, 'structure'), exist_ok=True)
        if style == True:
            os.makedirs(os.path.join(sample_dir, 'style'), exist_ok=True)
        sample_images_npy(sample_dir, config_path, ckpt_path, n_samples=sample_number, ddim_steps=200, ddim_eta=1., structure=structure, style=style, experiment=experiment)
        print(os.path.splitext(ckpt_path)[1], 'Finished\n')

def split_mode_folder(original_dir, result_dir):
    modes = ['T1', 'T2', 'FLAIR', 'WB', 'BB']
    for m in modes:
        os.makedirs(os.path.join(result_dir, m), exist_ok=True)
    all_original_filepath = glob.glob(original_dir+'/*.npy')
    for f in all_original_filepath:
        filename = os.path.basename(f)
        mode = filename.split('_')[0]
        print(os.path.join(result_dir, mode, filename))
        shutil.copyfile(f, os.path.join(result_dir, mode, filename))
    print(result_dir.split('/')[-1], 'finished')

if __name__ == '__main__':

    config_path = 'configs/latent-diffusion/disentangled_meta-ldm-kl-64x64x12.yaml'


    # Sample generation #####################################################################################################

    # ckpt_path = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/logs/epoch=000191.ckpt'
    # sample_dir = 'final_samples_191'
    # configure(config_path, ckpt_path, sample_dir, 1000)

    # ckpt_path = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/logs/epoch=000195.ckpt'
    # sample_dir = 'final_samples_195'
    # configure(config_path, ckpt_path, sample_dir, 1000)

    # ckpt_path = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/logs/epoch=000202.ckpt'
    # sample_dir = 'final_samples_202' ##
    # configure(config_path, ckpt_path, sample_dir, 1000)

    # ckpt_path = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/logs/epoch=000209.ckpt'
    # sample_dir = 'final_samples_209'
    # configure(config_path, ckpt_path, sample_dir, 1000)

    # split to modality folder###############################################################################################
    
    # original_dir ='/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/final_samples_202/images'
    # result_dir = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/1000_sample_epoch_202'
    # split_mode_folder(original_dir, result_dir)

    # original_dir ='/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/final_samples_191/images'
    # result_dir = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/1000_sample_epoch_191'
    # split_mode_folder(original_dir, result_dir)

    # original_dir ='/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/final_samples_195/images'
    # result_dir = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/1000_sample_epoch_195'
    # split_mode_folder(original_dir, result_dir)

    # original_dir ='/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/final_samples_209/images'
    # result_dir = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/1000_sample_epoch_209'
    # split_mode_folder(original_dir, result_dir)


    # visualize experiment #####################################################################################################

    # ckpt_path = '/home/yhn/Meta_synthesis/Project/models/DLDM/logs/2023-02-21T07-46-47_DLDM-kl-64x64x12/checkpoints/DLDM_epoch=000202.ckpt'
    # sample_dir = 'structure_style_experiment_202_230225'
    # configure(config_path, ckpt_path, sample_dir, sample_number=1, structure=False, style=False, experiment=True, interpolate=False)

    #########################################################################################################################

    ckpt_path = '/home/yhn/Meta_synthesis/Project/models/DLDM/logs/2023-02-21T07-46-47_DLDM-kl-64x64x12/checkpoints/DLDM_epoch=000202.ckpt'
    sample_dir = 'structure_T1_interpolation_experiment_202_230228'
    configure(config_path, ckpt_path, sample_dir, sample_number=5, structure=False, style=False, experiment=False, interpolate_style=False, interpolate_structure=True)