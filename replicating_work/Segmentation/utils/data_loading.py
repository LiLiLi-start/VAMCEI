import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import nibabel as nib
import os
import albumentations as A
import matplotlib.pyplot as plt
from torchvision import transforms
import PIL

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.images = []
        self.masks = []
        self.transform = transforms.Compose(
            # transforms.Grayscale(num_output_channels=1),
            [transforms.Resize(size=256, interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(size=(192, 192)),
            # transforms.ToTensor()
            # transforms.Normalize(mean=0.5, std=0.5)
            ]
            
        )
        for image_name in sorted(os.listdir(self.images_dir)):
            # patient6_0.npy
            # patient6_C0_manual.nii.gz
            
            # lab33_slice1.nii.gz
            # img33_slice1_0.npy
            
            image_path = os.path.join(self.images_dir, image_name)
            mask_path = os.path.join(self.mask_dir, "lab"+image_name.split('_slice')[0].split('img')[-1]+'_slice'+image_name.split('_slice')[-1].split('_')[0]+'.nii.gz')
            
            # image = nib.load(image_path).get_fdata()
            image = np.load(image_path, allow_pickle=True)
            mask = nib.load(mask_path).get_fdata()
            # (256, 256, 10)
            deep = image.shape[0]
            for i in range(deep):
                image_i = image[i:i+1, :, :].copy()
                # image_i = torch.tensor(image_i)
                mask_i = mask[i, :, :].copy()
                
                # plt.imshow(mask_i)
                # plt.show()
                
                
                mask_i = np.where((mask_i==205), 1, mask_i)
                mask_i = np.where((mask_i==420), 2, mask_i)
                mask_i = np.where((mask_i==500), 3, mask_i)
                mask_i = np.where((mask_i==550), 4, mask_i)
                mask_i = np.where((mask_i==600), 5, mask_i)
                mask_i = np.where((mask_i==820), 0, mask_i)
                mask_i = np.where((mask_i==850), 0, mask_i)
                
                
                
                mask_i = Image.fromarray(mask_i)
                mask_i = self.transform(mask_i)
                
                mask_i = np.array(mask_i)
                # torch.Size([1, 192, 192])
                
                self.images.append(image_i)
                self.masks.append(mask_i)


    def __len__(self):
        return len(self.images)*5

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        idx = idx%(len(self.images))

        image = self.images[idx][0].copy()
        mask = self.masks[idx].copy()
        
        randx = np.random.randint(-16, 16)
        randy = np.random.randint(-16, 16)
        # image = image[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        # mask = mask[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        
        augmentation = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.Resize(height=192, width=192, p=1),
            # A.ElasticTransform(p=1, alpha=20, sigma=120 * 0.08, alpha_affine=120 * 0.03),
        ])
        augmentation_out = augmentation(image=image, mask=mask)
        image = augmentation_out['image']
        mask = augmentation_out['mask']
        
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)
        # plt.show()
        
        
        
        return {
            'image': torch.as_tensor(image).unsqueeze(0).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous()
        }

class BasicValDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.images = []
        self.masks = []
        self.names = []
        # mask处理
        self.transform_mask = transforms.Compose(
            [transforms.Resize(size=286, interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(size=(192, 192)),
            transforms.ToTensor()]
        )
        # image处理
        self.transform_image = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=286, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(size=(192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]
        )
        for mask_name in sorted(os.listdir(self.mask_dir)):
            # patient1_LGE.nii.gz
            # patient1_LGE_manual.nii.gz
            
            
            # myops_training_101_DE.nii.gz
            # myops_training_101_gd.nii.gz
            
            # img1_slice1.nii.gz
            # lab1_slice1.nii.gz
            mask_path = os.path.join(self.mask_dir, mask_name)
            image_path = os.path.join(self.images_dir, 'img'+mask_name.split('lab')[-1])
            name = os.path.join(self.images_dir, 'img'+mask_name.split('lab')[-1])
            
            image = nib.load(image_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            # (512, 512, 15)
            deep = image.shape[0]
            
            image_process = torch.zeros((deep, 192, 192))
            mask_process = torch.zeros((deep, 192, 192))
            
            for i in range(deep):
                image_i = image[i, :, :].copy()
                mask_i = mask[i, :, :].copy()
                
                # mask_i = np.where((mask_i==500), 1, mask_i)
                # mask_i = np.where((mask_i==600), 2, mask_i)
                # mask_i = np.where((mask_i==200), 3, mask_i)
                # mask_i = np.where((mask_i==1220), 3, mask_i)
                # mask_i = np.where((mask_i==2221), 3, mask_i)
                mask_i = np.where((mask_i==205), 1, mask_i)
                mask_i = np.where((mask_i==420), 2, mask_i)
                mask_i = np.where((mask_i==500), 3, mask_i)
                mask_i = np.where((mask_i==550), 4, mask_i)
                mask_i = np.where((mask_i==600), 5, mask_i)
                mask_i = np.where((mask_i==820), 0, mask_i)
                mask_i = np.where((mask_i==850), 0, mask_i)
                
                
                
                mask_i = Image.fromarray(mask_i)
                mask_i = self.transform_mask(mask_i)
                
                image_i = image_i/(np.max(image_i)-np.min(image_i))*255.0
                image_i = Image.fromarray(image_i)
                image_i = self.transform_image(image_i)
                
                image_process[i] = image_i[0]
                mask_process[i] = mask_i[0]
        
            self.images.append(image_process)
            self.masks.append(mask_process)
            self.names.append(name)
        
        

    def __len__(self):
        return len(self.images)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):

        image = self.images[idx].clone()
        # image = (image-np.min(image))/(np.max(image)-np.min(image))
        mask = self.masks[idx].clone()
        name = self.names[idx]
        
        return {
            'image': image.float().contiguous(),
            'mask': mask.long().contiguous(),
            'name':name
        }

class BasicDatasetC0(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        # /home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/c0_image
        # 
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.images = []
        self.masks = []
        self.transform_mask = transforms.Compose(
            # transforms.Grayscale(num_output_channels=1),
            [transforms.Resize(size=256, interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(size=(192, 192)),
            transforms.ToTensor()
            ]
            # transforms.Normalize(mean=0.5, std=0.5)
        )
        self.transform_image = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=256, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(size=(192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]
        )
        for image_name in sorted(os.listdir(self.images_dir)):
            # patient6_C0.nii.gz
            # patient6_C0_manual.nii.gz
            
            image_path = os.path.join(self.images_dir, image_name)
            mask_path = os.path.join(self.mask_dir, image_name.split('_')[0]+'_C0_manual.nii.gz')
            
            image = nib.load(image_path).get_fdata()
            # image = np.load(image_path, allow_pickle=True)
            mask = nib.load(mask_path).get_fdata()
            # (256, 256, 10)
            deep = image.shape[-1]
            for i in range(deep):
                image_i = image[:, :, i].copy()
                mask_i = mask[:, :, i].copy()
                
                image_i = (image_i-np.min(image_i))/(np.max(image_i)-np.min(image_i))*255.0
                image_i = Image.fromarray(image_i)
                image_i = self.transform_image(image_i)
                image_i = np.array(image_i)
                # (1, 192, 192)
                
                mask_i = np.where((mask_i==500), 1, mask_i)
                mask_i = np.where((mask_i==600), 2, mask_i)
                mask_i = np.where((mask_i==200), 3, mask_i)
                mask_i = Image.fromarray(mask_i)
                mask_i = self.transform_mask(mask_i)
                mask_i = np.array(mask_i)
                # (1, 192, 192)
                
                self.images.append(image_i[0])
                self.masks.append(mask_i[0])


    def __len__(self):
        return len(self.images)*5

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        idx = idx%(len(self.images))

        image = self.images[idx].copy()
        mask = self.masks[idx].copy()
        
        randx = np.random.randint(-16, 16)
        randy = np.random.randint(-16, 16)
        # image = image[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        # mask = mask[96+randx-80:96+randx+80, 96+randy-80:96+randy+80]
        
        augmentation = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.Resize(height=192, width=192, p=1),
            # A.ElasticTransform(p=1, alpha=20, sigma=120 * 0.08, alpha_affine=120 * 0.03),
        ])
        augmentation_out = augmentation(image=image, mask=mask)
        image = augmentation_out['image']
        mask = augmentation_out['mask']
        
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)
        # plt.show()
        
        
        
        return {
            'image': torch.as_tensor(image).unsqueeze(0).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous()
        }



class BasicDatasetValC0(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        # /home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/c0_image
        # 
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.images = []
        self.masks = []
        self.transform_mask = transforms.Compose(
            # transforms.Grayscale(num_output_channels=1),
            [transforms.Resize(size=256, interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(size=(192, 192)),
            transforms.ToTensor()
            ]
            # transforms.Normalize(mean=0.5, std=0.5)
        )
        self.transform_image = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=256, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(size=(192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]
        )
        for image_name in sorted(os.listdir(self.images_dir)):
            # patient6_C0.nii.gz
            # patient6_C0_manual.nii.gz
            
            image_path = os.path.join(self.images_dir, image_name)
            mask_path = os.path.join(self.mask_dir, image_name.split('_')[0]+'_C0_manual.nii.gz')
            
            image = nib.load(image_path).get_fdata()
            # image = np.load(image_path, allow_pickle=True)
            mask = nib.load(mask_path).get_fdata()
            # (256, 256, 10)
            deep = image.shape[-1]
            image_pro = torch.zeros((deep, 192, 192))
            mask_pro = torch.zeros((deep, 192, 192))
            for i in range(deep):
                image_i = image[:, :, i].copy()
                mask_i = mask[:, :, i].copy()
                
                image_i = (image_i-np.min(image_i))/(np.max(image_i)-np.min(image_i))*255.0
                image_i = Image.fromarray(image_i)
                image_i = self.transform_image(image_i)
                # image_i = np.array(image_i)
                # (1, 192, 192)
                
                mask_i = np.where((mask_i==500), 1, mask_i)
                mask_i = np.where((mask_i==600), 2, mask_i)
                mask_i = np.where((mask_i==200), 3, mask_i)
                mask_i = Image.fromarray(mask_i)
                mask_i = self.transform_mask(mask_i)
                # mask_i = np.array(mask_i)
                # (1, 192, 192)
                
                image_pro[i] = image_i[0]
                mask_pro[i] = mask_i[0]
                
                
            self.images.append(image_pro)
            self.masks.append(mask_pro)


    def __len__(self):
        return len(self.images)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):

        image = self.images[idx].clone()
        mask = self.masks[idx].clone()

        return {
            'image': image.float().contiguous(),
            'mask': mask.long().contiguous()
        }





class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
