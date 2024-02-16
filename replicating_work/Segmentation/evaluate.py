import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

def Hausdorff_compute(pred, groundtruth, spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1, 6, 5))
    surface_distance_results = np.zeros((1, 6, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(6):
        pred_i = (pred == i).astype(np.float32)
        if np.sum(pred_i) == 0:
            overlap_results[0, i, :] = 0
            surface_distance_results[0, i, :] = 0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue == i, ITKPred == i)
            overlap_results[0, i,
                            0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0, i,
                            1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0, i,
                            2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0, i,
                            3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0, i,
                            4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue == i, ITKPred == i)

            surface_distance_results[0, i,
                                     0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(
                ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(
                statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(
                ITKPred == i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred == i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(
                statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * \
                sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * \
                sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(
                seg2ref_distance_map)
            seg2ref_distances = list(
                seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(
                ref2seg_distance_map)
            ref2seg_distances = list(
                ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0, i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0, i, 2] = np.median(
                all_surface_distances)
            surface_distance_results[0, i, 3] = np.std(all_surface_distances)
            surface_distance_results[0, i, 4] = np.max(all_surface_distances)

    return overlap_results, surface_distance_results

@torch.inference_mode()
def evaluate(net, net_2, dataloader, device, amp):
    net.eval()
    net_2.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            
            mask_pred_2 = net_2(torch.cat((mask_pred, image), dim=1))
            
            # mask_pred_ = torch.argmax(mask_pred[0], 0)
            # plt.imshow(mask_pred_.cpu().detach().numpy())
            # plt.show()

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred_2 = (F.sigmoid(mask_pred_2) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred_2, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_2 = F.one_hot(mask_pred_2.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred_2[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    net_2.train()
    return dice_score / max(num_val_batches, 1)


import time
@torch.inference_mode()
def evaluate_lge(net, net_2, dataloader, device, amp):
    net.eval()
    net_2.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # dice_score = 0
    time_list = []
    time_num = 0
    mean_dice_3 = np.zeros((4))
    mean_dice_list = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            deep = image.shape[1]
            output = torch.zeros((1, deep, 192, 192))
            # torch.Size([1, 15, 192, 192])
            # torch.Size([1, 15, 192, 192])
            
            deep = image.shape[1]
            for i in range(deep):
                image_i = image[:, i:i+1, :, :]
                
                start = time.time()
                mask_pred = net(image_i)
                mask_pred_2 = net_2(torch.cat((mask_pred, image_i), dim=1))
                end = time.time()
                
                time_list.append(end-start)
                time_num += 1
                # torch.Size([1, 4, 192, 192])
                mask_pred_2 = torch.argmax(mask_pred_2, dim=1)
                output[:, i] = mask_pred_2
            # 计算Dice
            dice = dice_compute(output.cpu().numpy(), mask_true.cpu().numpy())
            mean_dice_3 += dice
            print('dice={}'.format(dice))
            mean_dice = np.mean(dice[1:])
            print('mean dice={}'.format(mean_dice))
            mean_dice_list.append(mean_dice)
            
            dice_score += mean_dice
        
        time_all = 0
        for i in range(len(time_list)):
            time_all += time_list[i]

        print(1/(time_all/time_num))
        print(mean_dice_3/num_val_batches)
        print(mean_dice_list)

    net.train()
    net_2.train()
    return dice_score / max(num_val_batches, 1)

@torch.inference_mode()
def evaluate_lge_unet_1(net, dataloader, device, amp):
    net.eval()
    # net_2.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    mean_dice_list = []
    mean_dice_3 = np.zeros((6))
    total_surface_distance=np.zeros((1, 6, 5))
    # dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, name = batch['image'], batch['mask'], batch['name']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            deep = image.shape[1]
            output = torch.zeros((1, deep, 192, 192))
            # torch.Size([1, 15, 192, 192])
            # torch.Size([1, 15, 192, 192])
            
            deep = image.shape[1]
            for i in range(deep):
                image_i = image[:, i:i+1, :, :]
                
                mask_pred = net(image_i)
                # mask_pred_2 = net_2(torch.cat((mask_pred, image_i), dim=1))
                # torch.Size([1, 4, 192, 192])
                # (0.1, 0.1, 0.7 ,0.1)
                
                # plt.subplot(2, 2, 1)
                # plt.imshow(mask_pred[0, 0].cpu().detach().numpy())
                # plt.subplot(2, 2, 2)
                # plt.imshow(mask_pred[0, 1].cpu().detach().numpy())
                # plt.subplot(2, 2, 3)
                # plt.imshow(mask_pred[0, 2].cpu().detach().numpy())
                # plt.subplot(2, 2, 4)
                # plt.imshow(mask_pred[0, 3].cpu().detach().numpy())
                # plt.show()
                
                
                
                mask_pred = torch.argmax(mask_pred, dim=1)
                
                # plt.imshow(mask_pred[0].cpu().detach().numpy())
                # plt.show()
                
                
                output[:, i] = mask_pred
                
                # torch.Size([1, 5, 192, 192])
                # plt.subplot(1, 2, 1)
                # plt.imshow(image_i[0, 0].cpu().detach().numpy())
                # plt.subplot(1, 2, 2)
                # plt.imshow(mask_true[0, i].cpu().detach().numpy())
                # plt.show()
                
                
                
            # 计算Dice
            dice = dice_compute(output.cpu().numpy(), mask_true.cpu().numpy())
            # mean_dice_3 += dice
            mean_dice_3 = np.vstack((mean_dice_3, dice))
            print('dice={}'.format(dice))
            mean_dice = np.mean(dice[1:])
            dice_score += mean_dice
            mean_dice_list.append(mean_dice)
            
            name = name[0]
            overlap_result, surface_distance_result = Hausdorff_compute(output.cpu().numpy(),mask_true.cpu().numpy(),sitk.ReadImage(name).GetSpacing())
            total_surface_distance = np.concatenate((total_surface_distance,surface_distance_result),axis=0)
        
        meanDice = np.mean(mean_dice_3[1:], axis=0)
        stdDice = np.std(mean_dice_3[1:], axis=0)
        print(meanDice)
        print('----------------------')
        print(stdDice)
        print('----------------------')
        
        print(mean_dice_list)
        
        mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
        std_surface_distance = np.std(total_surface_distance[1:], axis=0)
        print('----------------------')
        print(mean_surface_distance)
        print('----------------------')
        print(std_surface_distance)

    net.train()
    # net_2.train()
    return dice_score / max(num_val_batches, 1)

def dice_compute(pred, groundtruth):  # batchsize*channel*W*W
    # for j in range(pred.shape[0]):
    #     for i in range(pred.shape[1]):
    #         if np.sum(pred[j,i,:,:])==0 and np.sum(groundtruth[j,i,:,:])==0:
    #             pred[j, i, :, :]=pred[j, i, :, :]+1
    #             groundtruth[j, i, :, :]=groundtruth[j,i,:,:]+1
    #
    # dice = 2*np.sum(pred*groundtruth,axis=(2,3),dtype=np.float16)/(np.sum(pred,axis=(2,3),dtype=np.float16)+np.sum(groundtruth,axis=(2,3),dtype=np.float16))
    dice = []
    for i in range(6):
        dice_i = 2*(np.sum((pred == i)*(groundtruth == i), dtype=np.float32)+0.0001) / \
            (np.sum(pred == i, dtype=np.float32) +
             np.sum(groundtruth == i, dtype=np.float32)+0.0001)
        dice = dice+[dice_i]

    return np.array(dice, dtype=np.float32)