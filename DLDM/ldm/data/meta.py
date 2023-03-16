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
            self.modes = ['T1', 'T2', 'FLAIR', 'WB', 'BB']
            if self.model == 'ldm':
                self.label_to_idx = dict((mode, i) for i, mode in enumerate(self.modes))
                self.idx_to_label = {v:k for k, v in self.label_to_idx.items()}
                self.labels["class_label"] = np.random.randint(0, 5, len(self.image_paths)) # 0, 1, 2, 3, 4 random sample
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
        if self.multi:
            image = image.transpose((1, 2, 0)) # only for multi sequences
        example["image"] = (image/0.5 - 1.0).astype(np.float32)
        # example["target_index"] = random.randint(1, 5)
        return example

class DAEMetaMultiTrain(MetaBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/yhn/Meta_synthesis/Project/models/DLDM/data/metastasis/meta_multi_without_mask_train.txt", 
                         data_root="/home/yhn/Meta_synthesis/data/Severance_multi_without_label/train", 
                         model="dae", multi=True, **kwargs)

class DAEMetaMultiValid(MetaBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/yhn/Meta_synthesis/Project/models/DLDM/data/metastasis/meta_multi_without_mask_valid.txt", 
                         data_root="/home/yhn/Meta_synthesis/data/Severance_multi_without_label/valid", 
                         model="dae", multi=True, **kwargs)


class LDMMetaMultiTrain(MetaBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/yhn/Meta_synthesis/Project/models/DLDM/data/metastasis/meta_multi_without_mask_train.txt", 
                         data_root="/home/yhn/Meta_synthesis/data/Severance_multi_without_label/train", 
                         model="ldm", multi=True, **kwargs)

class LDMMetaMultiValid(MetaBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="/home/yhn/Meta_synthesis/Project/models/DLDM/data/metastasis/meta_multi_without_mask_valid.txt", 
                         data_root="/home/yhn/Meta_synthesis/data/Severance_multi_without_label/valid", 
                         model="ldm", multi=True, **kwargs)
