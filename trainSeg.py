from turtle import st
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.utils.data import Dataset
from utils_for_transfer import *
import tqdm
import model.seg as seg
import model.unet as unet
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置超参数
STEP = 500
# 100 200 400 800
dataset_dir = '/home/hfcui/cmrseg2019_project/VarDA/Dataset/Patch192'
BatchSize = 6
WORKERSNUM = 10
EPOCH = 80
LR = 0.00005
WEIGHT_DECAY = 1e-8
model_path = '/home/hfcui/cmrseg2019_project/VarDA/save_model/'

def dice_compute(pred, groundtruth):  # batchsize*channel*W*W
    # for j in range(pred.shape[0]):
    #     for i in range(pred.shape[1]):
    #         if np.sum(pred[j,i,:,:])==0 and np.sum(groundtruth[j,i,:,:])==0:
    #             pred[j, i, :, :]=pred[j, i, :, :]+1
    #             groundtruth[j, i, :, :]=groundtruth[j,i,:,:]+1
    #
    # dice = 2*np.sum(pred*groundtruth,axis=(2,3),dtype=np.float16)/(np.sum(pred,axis=(2,3),dtype=np.float16)+np.sum(groundtruth,axis=(2,3),dtype=np.float16))
    dice = []
    for i in range(4):
        dice_i = 2*(np.sum((pred == i)*(groundtruth == i), dtype=np.float32)+0.0001) / \
            (np.sum(pred == i, dtype=np.float32) +
             np.sum(groundtruth == i, dtype=np.float32)+0.0001)
        dice = dice+[dice_i]

    return np.array(dice, dtype=np.float32)


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, val=False):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = torch.FloatTensor(input.shape[1])
    dice = dice.cuda()
    for channel in range(input.shape[1]):
        t = dice_coeff(input[:, channel, ...], target[:,
                       channel, ...], reduce_batch_first, epsilon)
        dice[channel] = t
    if val:
        return dice
    return torch.mean(dice)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def id2trainId(label):
    shape = label.shape
    # print(shape)
    results_map = torch.zeros((shape[0], 4, shape[1], shape[2]))
    results_map = results_map.cuda()

    LV = (label == 1)
    RV = (label == 2)
    MY = (label == 3)

    background = torch.logical_not(LV + RV + MY)

    results_map[:, 0, :, :] = torch.where(background, 1, 0)
    results_map[:, 1, :, :] = torch.where(LV, 1, 0)
    results_map[:, 2, :, :] = torch.where(RV, 1, 0)
    results_map[:, 3, :, :] = torch.where(MY, 1, 0)

    return results_map


# 加载GaussianDiffusion预训练模型
model = Unet(
    channels=1,
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=160,
    timesteps=1000,   # number of steps
    loss_type='l1',   # L1 or L2
    objective='pred_x0'
)
diffusion.model.load_state_dict(torch.load(
    '/home/hfcui/cmrseg2019_project/VarDA/save_model/GaussianDiffusion_Unet_epoch_18.pth')['model_state_dict'])
diffusion.eval()
diffusion = diffusion.cuda()

# 创建分割网络
# model_seg = seg.Segmentor(961, num_classes=4, num_layers=9)

model_seg = unet.UNet(n_channels = 961,  n_classes=4)

model_seg.cuda()
model_seg.train()
optimizer = torch.optim.Adam(
    model_seg.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()


# 创建数据集
SourceData = C0_TrainSet(dataset_dir, 30)
SourceData_loader = DataLoader(SourceData, batch_size=BatchSize,
                               shuffle=True, num_workers=WORKERSNUM, pin_memory=True, drop_last=True)

SourceData_val = C0_ValSet(dataset_dir, 5)
SourceData_val_loader = DataLoader(SourceData_val, batch_size=1, num_workers=WORKERSNUM, pin_memory=True)

# 训练
loss_epoch = 0.0
dice_val_best = 0.0
for i in range(EPOCH):
    loss_epoch = 0.0
    for indx, data in enumerate(SourceData_loader):
        image, label = data
        image = image.cuda()
        label = label.cuda()

        # plt.subplot(1,2,1)
        # plt.imshow(image[0,0].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(label[0].cpu().detach().numpy())
        # plt.show()
        
        optimizer.zero_grad()
        _, features = diffusion.forward_pretrain(image, step=STEP)
        predict = model_seg(features)

        label_one_hot = id2trainId(label=label)

        loss_criterion = criterion(predict.float(), label)
        loss_dice = dice_loss(
            F.softmax(predict, dim=1).float(), label_one_hot.float(), multiclass=True)
        loss = loss_criterion + loss_dice
        loss_epoch += loss
        loss.backward()
        optimizer.step()
        print('loss = {:.4f}'.format(loss))

    loss_epoch /= (indx+1)
    print('{} epoch: loss = {}'.format(i, loss_epoch))

    # 验证并保存best model
    # val
    if i % 1 == 0:
        with torch.no_grad():
            model_seg.eval()
            dice_val = np.zeros((4))
            for indx, data in enumerate(SourceData_val_loader):
                image_all, label_all = data
                image_all = image_all.cuda()
                label_all = label_all.cuda()
                deep = label_all.shape[1]

                # print(label_all.shape)
                # print(image_all.shape)
                # torch.Size([1, 12, 160, 160])
                # torch.Size([1, 1, 12, 160, 160])

                # torch.Size([12, 160, 160])
                predict_all = torch.zeros((label_all.shape[1],label_all.shape[2],label_all.shape[3]))

                for j in range(deep):
                    image = image_all[:, :, j, :, :]
                    label = label_all[:, j, :, :]

                    # plt.subplot(1,2,1)
                    # plt.imshow(image[0,0].cpu().detach().numpy())
                    # plt.subplot(1,2,2)
                    # plt.imshow(label[0].cpu().detach().numpy())
                    # plt.show()


                    _, features = diffusion.forward_pretrain(image, step=STEP)
                    predict = model_seg(features)
                    predict = torch.argmax(predict[0],dim=0)
                    predict_all[j] = predict
                

                dice = dice_compute(predict_all.cpu().detach().numpy(), label_all[0].cpu().detach().numpy())
                print(dice)
                dice_val += dice
            dice_val /= 5
            print('-----------------------------')
            print(dice_val)
            print('-----------------------------')
            if (np.sum(dice_val[1:])/3) > dice_val_best:
                dice_val_best = (np.sum(dice_val[1:])/3)
                print('save model...')
                print('dice_val_best = {:.4f}'.format(dice_val_best))
                torch.save(model_seg.state_dict(), model_path +
                           'seg_model_unet_best'+str(i)+'.pth')
            model_seg.train()
