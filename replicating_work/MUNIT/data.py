"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import nibabel as nib
import numpy as np
def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # if is_image_file(fname):
            path = os.path.join(root, fname)
            images.append(path)

    return images

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        # print(transform)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        # self.save_dir = '/home/hfcui/MUNIT/mscmrseg_root/generate_lge'
        # self.loader = loader
        # img_data = np.zeros((1, 256, 256))
        img_data = []
        # save_path = []
        for i in range(len(imgs)):
            img_data_i = nib.load(imgs[i]).get_fdata()

            # img_data_i = img_data_i.transpose(2, 0, 1)
            # print(img_data_i.shape)
            # (220, 1, 240)
            # img_data_i_transform = torch.zeros((img_data_i.shape[0], 192, 192))
            # (10, 256, 256)
            for j in range(img_data_i.shape[0]):
                img_data_i_j = img_data_i[j].copy()
                # img_data_i_j = (img_data_i_j-np.min(img_data_i_j))/(np.max(img_data_i_j)-np.min(img_data_i_j))*255.0
                # img_data_i_j = Image.fromarray(img_data_i_j)
                # if self.transform is not None:
                #     img_data_i_j = self.transform(img_data_i_j)
                if np.max(img_data_i_j)-np.min(img_data_i_j) < 1e-3:
                    continue
                img_data.append(img_data_i_j)
            # save_path.append(save_path_i)
            
        self.img_data = img_data


    def __getitem__(self, index):
        img = self.img_data[index]
        img = (img-np.min(img))/(np.max(img)-np.min(img))*255.0
        img = Image.fromarray(img)
        img = self.transform(img)
        return img


    def __len__(self):
        return len(self.img_data)



class ImageFolderTest(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.save_dir = '/home/hfcui/MUNIT/mmwhs_root/generate_mr_5'
        # self.loader = loader
        # img_data = np.zeros((1, 256, 256))
        img_data = []
        save_path = []
        for i in range(len(imgs)):
            img_data_i = nib.load(imgs[i]).get_fdata()
            # img33_slice1.nii.gz
            name = imgs[i].split('/')[-1].split('.nii.gz')[0]+'_'
            save_path_i = os.path.join(self.save_dir, name)
            save_path.append(save_path_i)

            # img_data_i = img_data_i.transpose(2, 0, 1)
            img_data_i_transform = torch.zeros((img_data_i.shape[0], 192, 192))
            # (10, 256, 256)
            for j in range(img_data_i.shape[0]):
                img_data_i_j = img_data_i[j].copy()
                if np.max(img_data_i_j)-np.min(img_data_i_j) > 1e-3:
                    img_data_i_j = (img_data_i_j-np.min(img_data_i_j))/(np.max(img_data_i_j)-np.min(img_data_i_j))*255.0
                img_data_i_j = Image.fromarray(img_data_i_j)
                if self.transform is not None:
                    img_data_i_j = self.transform(img_data_i_j)
                img_data_i_transform[j:j+1] = img_data_i_j
            img_data.append(img_data_i_transform)
            # save_path.append(save_path_i)
            
        self.img_data = img_data
        self.save_path = save_path
        # img_data = img_data[1:]


    def __getitem__(self, index):
        
        # img = self.img_data[index]
        # # img = torch.Tensor(img)
        # img = (img-np.min(img))/(np.max(img)-np.min(img))*255.0
        # img = Image.fromarray(img)
        # if self.transform is not None:
        #     img = self.transform(img)
        # print(img.shape)
        print(self.transform)
        
        # Compose(
        #     Grayscale(num_output_channels=1)
        #     Resize(size=256, interpolation=bilinear, max_size=None, antialias=warn)
        #     RandomCrop(size=(192, 192), padding=None)
        #     ToTensor()
        #     Normalize(mean=0.5, std=0.5)
        # )
        
        img = self.img_data[index]
        save_path = self.save_path[index]
        return img, save_path
    
        # path = self.imgs[index]
        # img = self.loader(path)
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.return_paths:
        #     return img, path
        # else:
        #     return img

    def __len__(self):
        return len(self.img_data)
