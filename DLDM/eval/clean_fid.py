from cleanfid import fid
import os 

# modes = ['T1', 'T2', 'FLAIR', 'WB', 'BB']
# real_root = '/home/yhn/Meta_synthesis/data/Severance_FID'
# fake_root = '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/1000_sample_epoch_202'
# for mode in modes:
#     real_path = os.path.join(real_root, mode)
#     fake_path = os.path.join(fake_root, mode)
#     score = fid.compute_fid(real_path, fake_path)
#     print(mode, ':', score)

modes = ['T1', 'T2', 'FLAIR', 'WB', 'BB']
real_root = '/home/yhn/Meta_synthesis/data/Severance_FID'
fake_root = '/home/yhn/Meta_synthesis/data/stylegan2'
for mode in modes:
    real_path = os.path.join(real_root, mode)
    fake_path = os.path.join(fake_root, mode)
    score = fid.compute_fid(real_path, fake_path, dataset_res=256, num_gen=1000)
    print(mode, ':', score)
