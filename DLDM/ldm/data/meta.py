import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

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

class MetaBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 model='ae',
                 multi=False,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        self.model = model
        self.multi = multi
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        if self.multi:
            self.modes = ['glio_T2', 'glio_FLAIR', 'glio_WB', 'glio_BB', 'glio_mask', 'meta_T2', 'meta_FLAIR', 'meta_WB', 'meta_BB', 'meta_mask']

            if self.model == 'ldm':
                self.label_to_idx = dict((mode, i) for i, mode in enumerate(self.modes))
                self.idx_to_label = {v:k for k, v in self.label_to_idx.items()}
                glio_meta_select = []
                for paths in self.labels["file_path_"]:
                    if len(os.path.basename(paths).split('_'))==3: # severance
                        glio_meta_select.append(np.random.randint(5, 10))
                    else:
                        glio_meta_select.append(np.random.randint(0, 5))

                self.labels["class_label"] = np.array(glio_meta_select) # 0 ~ 10 random sample: 10종류 도메인 중 생성할 영상 랜덤 픽
                # self.labels["class_label"] = np.random.randint(0, 5, len(self.image_paths)) # 0 ~ 5 random sample: 5종류 도메인 중 생성할 영상 랜덤 픽
                self.labels["human_label"] = np.array([self.idx_to_label[x] for x in self.labels["class_label"]])

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = np.load(example["file_path_"])
        mask = image[4, :, :]
        mask = np.where(mask > 0.875, 1, mask) # necro
        mask = np.where((mask > 0.625) & (mask < 0.875), 0.75, mask) # enhance
        mask = np.where((mask > 0.375) & (mask < 0.625), 0.5, mask) # edema
        mask = np.where((mask > 0.125) & (mask < 0.375), 0.25, mask) # meta
        mask = np.where(mask < 0.125, 0, mask) # background
        image[4, :, :] = mask
        

        if self.multi:
            image = image.transpose((1, 2, 0)) # only for multi sequences

        example["image"] = (image/0.5 - 1.0).astype(np.float32)
        return example

# class Trainset(MetaBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="/home/yhn/v3_DLDM_multi-institute_single_mask/DLDM/data/bb_imputation_train.txt", 
#                          data_root="/home/yhn/Meta_synthesis/data/bb_imputation/train", 
#                          model="dae", multi=True, **kwargs)

# class Validset(MetaBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="/home/yhn/v3_DLDM_multi-institute_single_mask/DLDM/data/bb_imputation_valid.txt", 
#                          data_root="/home/yhn/Meta_synthesis/data/bb_imputation/valid", 
#                          model="dae", multi=True, **kwargs)

# class Trainset(MetaBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="/home/yhn/v3_DLDM_multi-institute_single_mask/DLDM/data/bb_imputation_train.txt", 
#                          data_root="/home/yhn/Meta_synthesis/data/bb_imputation/train", 
#                          model="ldm", multi=True, **kwargs)

# class Validset(MetaBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="/home/yhn/v3_DLDM_multi-institute_single_mask/DLDM/data/bb_imputation_valid.txt", 
#                          data_root="/home/yhn/Meta_synthesis/data/bb_imputation/valid", 
#                          model="ldm", multi=True, **kwargs)

class Trainset(MetaBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/your/path/to/train_textfile.txt", 
                         data_root="/your/path/to/dataset/train", 
                         model="ldm", multi=True, **kwargs)

class Validset(MetaBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/your/path/to/valid_textfile.txt", 
                         data_root="/your/path/to/dataset/valid", 
                         model="ldm", multi=True, **kwargs)
