import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate, evaluate_lge, evaluate_lge_unet_1
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, BasicValDataset
from utils.dice_score import dice_loss
import matplotlib.pyplot as plt

dir_img = Path('/home/hfcui/MUNIT/mscmrseg_root/generate_lge/')
dir_mask = Path('/home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/c0gt/')

dir_img_val = Path('/home/hfcui/MUNIT/mscmrseg_root/generate_lge/')
dir_mask_val = Path('/home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/c0gt_val/')

# dir_img_lge_val = Path('/home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/lge_image/')
# dir_mask_lge_val = Path('/home/hfcui/cmr2019_data/C0LET2_gt_for_challenge19/C0LET2_gt_for_challenge19/LGE_manual_35_TestData/')
# /home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/lge_image
# /home/hfcui/cmr2019_data/C0LET2_gt_for_challenge19/C0LET2_gt_for_challenge19/LGE_manual_35_TestData

dir_img_lge_val = Path('/home/hfcui/MUNIT/mmwhs_root/test_img/')
dir_mask_lge_val = Path('/home/hfcui/MUNIT/mmwhs_root/test_lab/')

# /home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/myops/lge_style_test_image
dir_checkpoint = Path('./checkpoints/')


def test_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        model_2=None
):


    val_set_lge = BasicValDataset(dir_img_lge_val, dir_mask_lge_val, img_scale)    
    
    n_val_lge = len(val_set_lge)
    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_lge_loader = DataLoader(val_set_lge, shuffle=False, drop_last=True, batch_size=1, num_workers=os.cpu_count(), pin_memory=True)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Validation LGE size: {n_val_lge}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    
    
    # val_score = evaluate_lge(model, model_2, val_lge_loader, device, amp)
    val_score = evaluate_lge_unet_1(model, val_lge_loader, device, amp)
    logging.info('Validation LGE Dice score: {}'.format(val_score))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=2e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default='/home/hfcui/Pytorch-UNet-master/mmwhs_checkpoint/checkpoints_unet1_lge/last.pth', help='Load model from a .pth file')
    parser.add_argument('--load_2', '-f2', type=str, default='/home/hfcui/Pytorch-UNet-master/checkpoints_unet2_lge/last_2.pth', help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes')
    parser.add_argument('--input_channel_unet1', '-i1', type=int, default=1, help='Number of input channel for unet1')
    parser.add_argument('--input_channel_unet2', '-i2', type=int, default=5, help='Number of input channel for unet2')

    return parser.parse_args()

import time
if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=args.input_channel_unet1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    
    # model_2 = UNet(n_channels=args.input_channel_unet1+args.classes, n_classes=args.classes, bilinear=args.bilinear)
    # model_2 = model_2.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    # logging.info(f'Network_2:\n'
    #              f'\t{model_2.n_channels} input channels\n'
    #              f'\t{model_2.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if model_2.bilinear else "Transposed conv"} upscaling')
    if args.load and args.load_2:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
        
        # state_dict = torch.load(args.load_2, map_location=device)
        # del state_dict['mask_values']
        # model_2.load_state_dict(state_dict)
        # logging.info(f'Model 2 loaded from {args.load_2}')
    model.to(device=device)
    # model_2.to(device=device)
    # try:
    
    test_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
        # model_2=model_2
    )
