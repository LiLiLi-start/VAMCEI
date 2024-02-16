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
from utils.edge_loss import edge_loss

# dir_img = Path('/home/hfcui/MUNIT/mscmrseg_root/generate_lge_5/')
# dir_mask = Path('/home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/c0gt/')

dir_img = Path('/home/hfcui/MUNIT/mmwhs_root/generate_mr_5/')
dir_mask = Path('/home/hfcui/MUNIT/mmwhs_root/CTgt/')


# dir_img_lge_val = Path('/home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/lge_val_image/')
# dir_mask_lge_val = Path('/home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/lge_val_gt/')
# /home/hfcui/cmr2019_data/C0LET2_nii45_for_challenge19/lge_image
# /home/hfcui/cmr2019_data/C0LET2_gt_for_challenge19/C0LET2_gt_for_challenge19/LGE_manual_35_TestData

dir_img_lge_val = Path('/home/hfcui/MUNIT/mmwhs_root/val_img/')
dir_mask_lge_val = Path('/home/hfcui/MUNIT/mmwhs_root/val_label/')


dir_checkpoint = Path('./mmwhs_checkpoint/checkpoints_unet1_lge/')


def train_model(
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
        # model_2=None
):
    # 1. Create dataset
    # 创建数据集
    try:
        train_set = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        train_set = BasicDataset(dir_img, dir_mask, img_scale)

    # try:
    #     val_set = CarvanaDataset(dir_img_val, dir_mask_val, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     val_set = BasicDataset(dir_img_val, dir_mask_val, img_scale)
    
    val_set_lge = BasicValDataset(dir_img_lge_val, dir_mask_lge_val, img_scale)    
    
    n_train = len(train_set)
    # n_val = len(val_set)
    n_val_lge = len(val_set_lge)
    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    val_lge_loader = DataLoader(val_set_lge, shuffle=False, drop_last=True, batch_size=1, num_workers=os.cpu_count(), pin_memory=True)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation LGE size: {n_val_lge}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    
    # [{'params': model.parameters()},{'params': model_2.parameters()}]
    
    # optimizer = optim.AdamW(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, foreach=True)
    
    optimizer = optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    val_score_max = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # torch.Size([8, 192, 192])
                        # torch.Size([8, 4, 192, 192])
                        # if torch.sum(true_masks[0])==0:
                            
                        #     print(true_masks.dtype)
                        #     print(torch.sum(true_masks[0]))
                        #     print(true_masks[0].cpu().detach().numpy().dtype)
                        #     plt.imshow(true_masks[0].cpu().detach().numpy())
                        #     plt.show()
                            
                        #     plt.imshow(torch.argmax(masks_pred[0], dim=0).cpu().detach().numpy())
                        #     plt.show()
                        #     print(F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).shape)
                        
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(images[0, 0].cpu().detach().numpy())
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(true_masks[0].cpu().detach().numpy())
                        # plt.show()
                        
                        edgeLoss = edge_loss(torch.argmax(masks_pred, dim=1, keepdim=True).float(), true_masks.unsqueeze(1).float())
                        loss = criterion(masks_pred, true_masks)
                        loss += edgeLoss*0.5
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )




                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})


        # val_score = evaluate(model, model_2, val_loader, device, amp)
        
        
        # 模型在真实的LGE上验证
        val_score = evaluate_lge_unet_1(model, val_lge_loader, device, amp)
        scheduler.step(val_score)
        logging.info('Validation Dice score: {}'.format(val_score))

        if save_checkpoint and val_score>val_score_max:
            val_score_max = val_score
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            torch.save(state_dict, str(dir_checkpoint / 'last.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)

    print('Total: {}, Trainable: {}'.format(
        total_num/(1024*1024), trainable_num/(1024*1024)))
    return total_num/(1024*1024)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default='/home/hfcui/Pytorch-UNet-master/checkpoints_unet1_c0/last.pth', help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes')
    parser.add_argument('--input_channel_unet1', '-i1', type=int, default=1, help='Number of input channel for unet1')
    parser.add_argument('--input_channel_unet2', '-i2', type=int, default=5, help='Number of input channel for unet2')

    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=args.input_channel_unet1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    
    
    model_num = get_parameter_number(model)
    print(model_num)
    
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
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    # model_2.to(device=device)
    # try:
    
    train_model(
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
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                   'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                   'Consider enabling AMP (--amp) for fast and memory efficient training')
    #     torch.cuda.empty_cache()
    #     model.use_checkpointing()
    #     train_model(
    #         model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #         img_scale=args.scale,
    #         val_percent=args.val / 100,
    #         amp=args.amp
    #     )
