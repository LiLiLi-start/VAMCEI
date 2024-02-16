from torch.cuda.amp import autocast as autocast
# import cv2
import random
from model_util.discriminator import FCDiscriminator
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils_for_transfer import *
# import scipy.misc
# import time
# import torchvision.transforms as transforms
# import torchvision
# from torch.autograd import Variable
from torch import optim
from torch.backends import cudnn
import torch.nn.functional as F
# from torch.nn import DataParallel
from torch.utils.data import DataLoader
import glob
import numpy as np
import SimpleITK as sitk
# import math
# from torch.utils.data import Dataset
from torch import nn
from sklearn import preprocessing
import torch
from queue import Queue,LifoQueue,PriorityQueue
# from turtle import forward
# from pyexpat import model
# from codecs import ignore_errors
from sklearn.manifold import TSNE
import pandas as pd
# import seaborn as sns
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#import nibabel as nib

# import matplotlib
# matplotlib.use('Agg')


EPOCH = 30
KLDLamda = 1.0

# PredLamda=1e3
# DisLamda=1e-4
LR = 1e-4
ADA_DisLR = 1e-4
LR_D = 5e-4
LR_G = 1e-5

WEIGHT_DECAY = 1e-5
WORKERSNUM = 10
dataset_dir = '/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA/Dataset/Patch192'
prefix = '/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA'
# TestDir=['/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192_Validation/','/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192/']
# TestDir=[dataset_dir+'/LGE_Test/',dataset_dir+'/LGE_Vali/']
# TestDir = ['/home/hfcui/cmrseg2019_project/VarDA/Dataset/Patch192/LGE_val/',]
        #    '/home/hfcui/cmrseg2019_project/VarDA/Dataset/Patch192/C0_val/']

TestDir = ['/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA/Dataset/Patch192/LGE_val/']


BatchSize = 4
KERNEL = 4
source_label = 0
target_label = 1
mode = 'Vanilla'
num_class = 4
# SAVE_DIR =prefix+ '/save_train_param'
# SAVE_IMG_DIR=prefix+'/save_test_label'


class h(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, v):
        if torch.sum(u.mul(u)) != 0 and torch.sum(v.mul(v)) != 0:
            return torch.sum(u.mul(v))/(torch.sum(u.mul(u))*torch.sum(v.mul(v)))
        else:
            return torch.sum(u.mul(v))


class prototype_g:
    def __init__(self, num_class, feat_dim, lam = 0.99) -> None:
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.pro_g = torch.zeros((num_class,feat_dim)).cuda()
        self.state = False
        # self.pixel = 0
        self.lamada = lam

    def update(self, feature, label):
        '''
        feature:torch.size(bs, dim, h, w)
        label:torch.size(bs, h, w)
        '''
        # label_k:torch.size(bs, dim, h, w)
        self.state = True
        feature_ = feature.clone()
        label_k = torch.unsqueeze(label.clone(), dim=1)
        label_k = label_k.repeat(1, self.feat_dim, 1, 1).cuda()
        # one = torch.zeros(label_k.shape).cuda()
        for i in range(self.num_class):
            feature_i = torch.where((label_k==i),feature_,0)
            # one_i = torch.where((label_k==i), one, 0)
            
            # 计算i类别像素数量和其特征均值
            # num_pixel = torch.sum(one_i)
            feature_i_ave = torch.sum(feature_i,dim=(0,2,3))
            # self.pro_g[i] = (self.pro_g[i]*self.pixel+feature_i_ave*num_pixel)/(self.pixel+num_pixel)
            if self.pro_g[i] == 0:
                self.pro_g[i] = feature_i_ave
            else:
                self.pro_g[i] = self.pro_g[i]*self.lamada+feature_i_ave*(1-self.lamada)
            # self.pixel += num_pixel
        
        return
    def update_clu(self, clu):

        for i in range(self.num_class):

            if self.state == False:
                self.pro_g[i] = clu[i]
                self.state = True
            else:
                self.pro_g[i] = self.pro_g[i]*self.lamada+clu[i]*(1-self.lamada)
            # self.pixel += num_pixel
        
        return
    
    def get_pro_g(self):

        return self.pro_g.clone()
    
    def get_pro_state(self):

        return self.state
    






# bs内部聚类中心对齐

def ADA_Train(Train_LoaderA, Train_LoaderB, encoder, decoderA, decoderAdown2, decoderAdown4, decoderB, decoderBdown2, decoderBdown4, gate, DistanceNet, lr, kldlamda, predlamda, dislamda, dislamdadown2, dislamdadown4, epoch, optim, savedir, model_D0, mode, writer):

    # if epoch >= 5:

    lr = lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr

    # 鉴别器损失函数
    if mode == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif mode == 'LS':
        bce_loss = torch.nn.MSELoss()

    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    max_iter = min(len(A_iter), len(B_iter))

    i = 0

    while i < len(A_iter) and i < len(B_iter):

        # torch.autograd.set_detect_anomaly(True)
        ct, ct_down2, ct_down4, label, label_down2, label_down4, info_ct = A_iter.next()
        mr, mr_down2, mr_down4, info_mr = B_iter.next()

        ct = ct.cuda()
        ct_down2 = ct_down2.cuda()
        ct_down4 = ct_down4.cuda()
        info_ct = info_ct.cuda()

        mr = mr.cuda()
        mr_down4 = mr_down4.cuda()
        mr_down2 = mr_down2.cuda()
        info_mr = info_mr.cuda()

        label = label.cuda()
        label_onehot = torch.FloatTensor(label.size(
            0), 4, label.size(1), label.size(2)).cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1, label.unsqueeze(dim=1), 1)

        # print(label_onehot.shape)
        # plt.subplot(2,2,1)
        # plt.imshow(label_onehot[0,0].cpu().detach().numpy())
        # plt.subplot(2,2,2)
        # plt.imshow(label_onehot[0,1].cpu().detach().numpy())
        # plt.subplot(2,2,3)
        # plt.imshow(label_onehot[0,2].cpu().detach().numpy())
        # plt.subplot(2,2,4)
        # plt.imshow(label_onehot[0,3].cpu().detach().numpy())
        # plt.show()

        label_down2 = label_down2.cuda()
        label_down2_onehot = torch.FloatTensor(label_down2.size(
            0), 4, label_down2.size(1), label_down2.size(2)).cuda()
        label_down2_onehot.zero_()
        label_down2_onehot.scatter_(1, label_down2.unsqueeze(dim=1), 1)

        label_down4 = label_down4.cuda()
        label_down4_onehot = torch.FloatTensor(label_down4.size(
            0), 4, label_down4.size(1), label_down4.size(2)).cuda()
        label_down4_onehot.zero_()
        label_down4_onehot.scatter_(1, label_down4.unsqueeze(dim=1), 1)

        fusionseg, _, out_ct, feat_ct, mu_ct, logvar_ct, _, outdown2_ct, featdown2_ct, mudown2_ct, logvardown2_ct, _, outdown4_ct, featdown4_ct, mudown4_ct, logvardown4_ct, info_pred_ct = encoder(
            ct, gate)

        # feat_ct --> z
        # shape torch.Size([4, 64, 160, 160])
        # label.shape --> torch.Size([4, 160, 160])
        label_k = torch.unsqueeze(label, dim=1)
        label_k = label_k.repeat(1, 64, 1, 1)
        # label_k = label_k.float()
        feat_ct = feat_ct.float()
        # torch.Size([4, 64, 160, 160])
        ignore = torch.zeros(feat_ct.shape)
        ignore = ignore.float()
        ignore = ignore.cuda()

        reserve = torch.ones(feat_ct.shape)
        reserve = reserve.float()
        reserve = reserve.cuda()

        clu_source = torch.zeros((num_class, 64))
        for k in range(num_class):

            # plt.imshow(feat_ct[0,0].cpu().detach().numpy())
            # plt.show()
            feat_ct_k = torch.where((label_k == k), feat_ct, ignore)

            # print(feat_ct_k.shape)
            # plt.imshow(feat_ct_k[0,0].cpu().detach().numpy())
            # plt.show()

            reserve_k = torch.where((label_k == k), reserve, ignore)
            feat_ct_k = torch.sum(feat_ct_k, dim=(0, 2, 3))
            reserve_k = torch.sum(reserve_k[:, 0], dim=(0, 1, 2))

            # print(reserve_k)
            # print(feat_ct_k)

            if reserve_k != 0:
                feat_ct_k = feat_ct_k/reserve_k
            else:
                feat_ct_k = feat_ct_k*0

            # feat_ct_k = torch.mean(feat_ct_k, dim=(0))

            # print(feat_ct_k)
            clu_source[k] = feat_ct_k

            # torch.Size([4, 64, 160, 160])
            # torch.Size([4, 64])
        clu_source = clu_source.cuda()
        # print(feat_ct.shape)
        # feat_ct torch.Size([4, 64, 160, 160])
        # feat_ct_contrastive = feat_ct.view((feat_ct.shape[1], feat_ct.shape[0]*feat_ct.shape[2]*feat_ct.shape[3])).transpose(1, 0)
        # clu_source_contrastive = clu_source.transpose(1, 0)

        # # 1. cycleGAN和VAE结合
        # # 2. 构造中间域
        # # 

        # # (1,4,num)
        # loss_contrastive = F.cross_entropy(
        #     feat_ct_contrastive.mm(clu_source_contrastive).transpose(0,1).unsqueeze(0),
        #     label.view((label.shape[0]*label.shape[1]*label.shape[2])).unsqueeze(0)
        # )
        # loss_contrastive = loss_contrastive*1000
        
        

        # print("loss_contrastive = {}".format(loss_contrastive.item()))
        # print(clu_source_contrastive.shape)








        #info_pred_ct = Infonet(info_pred_ct)

        info_cri = nn.CrossEntropyLoss().cuda()
        #infoloss_ct = info_cri(info_pred_ct,info_ct)
        seg_dice = DiceLoss()
        seg_dice = seg_dice.cuda()


        fusionsegloss_output_dice = seg_dice(fusionseg, label_onehot)

        seg_criterian = BalancedBCELoss(label)
        seg_criterian = seg_criterian.cuda()
        fusionsegloss_output_ce = seg_criterian(fusionseg, label)

        segloss_output = seg_criterian(
            out_ct, label) + seg_dice(fusionseg, label_onehot)
        # + seg_edge(out_ct, label_onehot, 4)*0.01


        fusionsegloss_output = seg_criterian(fusionseg, label) + seg_dice(
            fusionseg, label_onehot)
        # + seg_edge(fusionseg, label_onehot, 4)*0.01

        segdown2_criterian = BalancedBCELoss(label_down2)
        segdown2_criterian = segdown2_criterian.cuda()
        segdown2loss_output = segdown2_criterian(outdown2_ct, label_down2) + seg_dice(
            outdown2_ct, label_down2_onehot)
        # + seg_edge(outdown2_ct, label_down2_onehot, 4)*0.01

        segdown4_criterian = BalancedBCELoss(label_down4)
        segdown4_criterian = segdown4_criterian.cuda()
        segdown4loss_output = segdown4_criterian(outdown4_ct, label_down4) + seg_dice(
            outdown4_ct, label_down4_onehot)
        # + seg_edge(outdown4_ct, label_down4_onehot, 4)*0.01

        recon_ct = decoderA(feat_ct, label_onehot)
        BCE_ct = F.binary_cross_entropy_with_logits(recon_ct, ct)
        KLD_ct = -0.5 * torch.mean(1 + logvar_ct -
                                   mu_ct.pow(2) - logvar_ct.exp())

        recondown2_ct = decoderAdown2(featdown2_ct, label_down2_onehot)
        BCE_down2_ct = F.binary_cross_entropy_with_logits(
            recondown2_ct, ct_down2)
        KLD_down2_ct = -0.5 * \
            torch.mean(1 + logvardown2_ct -
                       mudown2_ct.pow(2) - logvardown2_ct.exp())

        recondown4_ct = decoderAdown4(featdown4_ct, label_down4_onehot)
        BCE_down4_ct = F.binary_cross_entropy_with_logits(
            recondown4_ct, ct_down4)
        KLD_down4_ct = -0.5 * \
            torch.mean(1 + logvardown4_ct -
                       mudown4_ct.pow(2) - logvardown4_ct.exp())

        fusionseg_mr, pred_mr, _, feat_mr, mu_mr, logvar_mr, preddown2_mr, _, featdown2_mr, mudown2_mr, logvardown2_mr, preddown4_mr, _, featdown4_mr, mudown4_mr, logvardown4_mr, info_pred_mr = encoder(
            mr, gate)
        

        label_target = torch.argmax(pred_mr, dim=1)

        label_k_target = torch.unsqueeze(label_target, dim=1)
        label_k_target = label_k_target.repeat(1, 64, 1, 1)
        # label_k = label_k.float()
        feat_ct = feat_ct.float()
        # torch.Size([4, 64, 160, 160])

        clu_target = torch.zeros((num_class, 64))

        for k in range(num_class):

            feat_mr_k = torch.where((label_k_target == k), feat_mr, ignore)
            reserve_mr_k = torch.where((label_k_target == k), reserve, ignore)

            feat_mr_k = torch.sum(feat_mr_k, dim=(0, 2, 3))
            reserve_mr_k = torch.sum(reserve_mr_k[:, 0], dim=(0, 1, 2))

            # print(reserve_mr_k)
            if reserve_mr_k != 0:
                feat_mr_k = feat_mr_k/reserve_mr_k
            else:
                feat_mr_k = feat_mr_k*0.0

            clu_target[k] = feat_mr_k

        clu_target = clu_target.cuda()




        # clu_source = clu_source.cuda()
        # print(feat_ct.shape)
        # feat_ct torch.Size([4, 64, 160, 160])
        # feat_mr_contrastive = feat_mr.view((feat_mr.shape[1], feat_mr.shape[0]*feat_mr.shape[2]*feat_mr.shape[3])).transpose(1, 0)
        # clu_target_contrastive = clu_target.transpose(1, 0)

        # # (1,4,num)
        # loss_contrastive_mr = F.cross_entropy(
        #     feat_mr_contrastive.mm(clu_target_contrastive).transpose(0,1).unsqueeze(0),
        #     label_target.view((label_target.shape[0]*label_target.shape[1]*label_target.shape[2])).unsqueeze(0)
        # )
        # loss_contrastive_mr = loss_contrastive_mr*1000
        
        

        # print("loss_contrastive_mr = {}".format(loss_contrastive_mr.item()))
            # torch.Size([4, 64, 160, 160])
            # torch.Size([4, 64])
        # if q_target.full():
        #     q_target.get()
        
        # q_target.put(clu_target)

        #infoloss_mr = info_cri(info_pred_mr,info_mr)

        recon_mr = decoderB(feat_mr, pred_mr)
        BCE_mr = F.binary_cross_entropy_with_logits(recon_mr, mr)
        KLD_mr = -0.5 * torch.mean(1 + logvar_mr -
                                   mu_mr.pow(2) - logvar_mr.exp())

        recondown2_mr = decoderBdown2(featdown2_mr, preddown2_mr)
        BCE_down2_mr = F.binary_cross_entropy_with_logits(
            recondown2_mr, mr_down2)
        KLD_down2_mr = -0.5 * \
            torch.mean(1 + logvardown2_mr -
                       mudown2_mr.pow(2) - logvardown2_mr.exp())

        recondown4_mr = decoderBdown4(featdown4_mr, preddown4_mr)
        BCE_down4_mr = F.binary_cross_entropy_with_logits(
            recondown4_mr, mr_down4)
        KLD_down4_mr = -0.5 * \
            torch.mean(1 + logvardown4_mr -
                       mudown4_mr.pow(2) - logvardown4_mr.exp())

        # print('-------------------------------')
        # print(torch.isnan(logvar_ct).any())
        # print(torch.isnan(mu_ct).any())
        # print(torch.isnan(logvar_mr).any())
        # print(torch.isnan(mu_mr).any())
        # print('-------------------------------')
        # if torch.isnan(logvar_ct).any():
        #     print(ct.shape)
        #     plt.imshow(ct[0,0].cpu().detach().numpy())
        #     plt.show()
        # print('-------------------------------')

        # distance_loss = DistanceNet(
        #     mu_ct, logvar_ct, mu_mr, logvar_mr) 
        # distance_down2_loss = DistanceNet(
        #     mudown2_ct, logvardown2_ct, mudown2_mr, logvardown2_mr)
        # distance_down4_loss = DistanceNet(
        #     mudown4_ct, logvardown4_ct, mudown4_mr, logvardown4_mr)



        distance_loss = DistanceNet(
            mu_ct, logvar_ct, mu_mr, logvar_mr) + DistanceNet(mu_mr, logvar_mr, mu_ct, logvar_ct)
        distance_down2_loss = DistanceNet(
            mudown2_ct, logvardown2_ct, mudown2_mr, logvardown2_mr) + DistanceNet(
            mudown2_mr, logvardown2_mr, mudown2_ct, logvardown2_ct)
        distance_down4_loss = DistanceNet(
            mudown4_ct, logvardown4_ct, mudown4_mr, logvardown4_mr) + DistanceNet(
            mudown4_mr, logvardown4_mr, mudown4_ct, logvardown4_ct)

        # 源域和目标域相同类别的原型对齐
        #
        clu_loss = torch.zeros((4,))
        for k in range(num_class):
            clu_k_positive = torch.sum(clu_source[k].mul(clu_target[k]))
            # clu_k_positive = clu_mseloss(clu_source[k],clu_target[k])
            clu_k_negative = torch.Tensor([0.0]).cuda()
            for l in range(num_class):
                if k != l:
                    if clu_k_negative == None:
                        # clu_k_negative = torch.exp(clu_mseloss(clu_source[l],clu_target[k]))
                        # clu_k_negative += torch.exp(clu_mseloss(clu_target[l],clu_target[k]))

                        clu_k_negative = torch.exp(
                            torch.sum(clu_source[l].mul(clu_target[k])))
                        clu_k_negative += torch.exp(
                            torch.sum(clu_target[l].mul(clu_target[k])))
                    else:
                        # clu_k_negative += torch.exp(clu_mseloss(clu_source[l],clu_target[k]))
                        # clu_k_negative += torch.exp(clu_mseloss(clu_target[l],clu_target[k]))
                        clu_k_negative += torch.exp(
                            torch.sum(clu_source[l].mul(clu_target[k])))
                        clu_k_negative += torch.exp(
                            torch.sum(clu_target[l].mul(clu_target[k])))

            clu_loss[k] = -torch.log(torch.exp(clu_k_positive) /
                                     (clu_k_negative+torch.exp(clu_k_positive)))

        clu_loss = torch.mean(clu_loss)
        clu_loss = clu_loss * 1000

        # # 收紧目标域与源域特征相对于类原型的分布
        # # 1.图像级别

        # # feat_ct 源域特征
        # # torch.Size([4, 64, 160, 160])

        # # feat_mr 目标域特征
        # # torch.Size([4, 64, 160, 160])

        # # clu_source源域各类别原型
        # # torch.zeros((num_class, 64))

        # # label_k源域标签
        # # torch.Size([4, 64, 160, 160])

        # # 1.1分别计算源域和目标域中 图像级别的原型

        # clu_image_source = torch.zeros((num_class, BatchSize, 64))
        # clu_image_target = torch.zeros((num_class, BatchSize, 64))

        # for k in range(num_class):
        #     feat_ct_image_k = torch.where((label_k == k), feat_ct, ignore)
        #     feat_ct_image_k = torch.sum(feat_ct_image_k, dim=(2, 3))
        #     # torch.Size([BatchSize, 64, ])

        #     reserve_image_ct_k = torch.where((label_k == k), reserve, ignore)
        #     reserve_image_ct_k = torch.sum(
        #         reserve_image_ct_k[:, 0], dim=(1, 2))
        #     # torch.Size([BatchSize, ])

        #     # 目标域
        #     feat_mr_image_k = torch.where(
        #         (label_k_target == k), feat_mr, ignore)
        #     feat_mr_image_k = torch.sum(feat_mr_image_k, dim=(2, 3))
        #     # torch.Size([BatchSize, 64, ])

        #     reserve_image_mr_k = torch.where(
        #         (label_k_target == k), reserve, ignore)
        #     reserve_image_mr_k = torch.sum(
        #         reserve_image_mr_k[:, 0], dim=(1, 2))
        #     # torch.Size([BatchSize, ])

        #     for l in range(BatchSize):
        #         if reserve_image_ct_k[l] != 0:
        #             feat_ct_image_k[l] = feat_ct_image_k[l] / \
        #                 reserve_image_ct_k[l]
        #         else:
        #             feat_ct_image_k[l] = feat_ct_image_k[l]*0

        #     clu_image_source[k] = feat_ct_image_k[l]

        #     for l in range(BatchSize):
        #         if reserve_image_mr_k[l] != 0:
        #             feat_mr_image_k[l] = feat_mr_image_k[l] / \
        #                 reserve_image_mr_k[l]
        #         else:
        #             feat_mr_image_k[l] = feat_mr_image_k[l]*0

        #     clu_image_target[k] = feat_mr_image_k[l]

        # clu_loss_image = torch.zeros((4,))
        # # 1.2拉近源域和目标域同类原型距离 推远不同类别原型距离
        # for k in range(num_class):
        #     index = random.randint(0, BatchSize-1)
        #     anchor = clu_image_source[k, index]
        #     index_target = random.randint(0, BatchSize-1)
        #     positive = clu_image_target[k, index_target]
        #     anchor_positive = torch.exp(torch.sum(anchor.mul(positive)))
        #     anchor_negative = torch.Tensor([0.0]).cuda()
        #     for l in range(num_class):
        #         if l != k:
        #             for m in range(BatchSize):
        #                 anchor_negative += torch.exp(
        #                     torch.sum(anchor.mul(clu_image_source[l, m])))
        #                 anchor_negative += torch.exp(
        #                     torch.sum(anchor.mul(clu_image_target[l, m])))

        #     clu_loss_image[k] = -torch.log(anchor_positive /
        #                                    (anchor_negative+anchor_positive))
        # clu_loss_image = torch.mean(clu_loss_image)

        # # 2.像素级别

        out_target = model_D0(fusionseg_mr)
        loss_adv_target0 = bce_loss(out_target, torch.FloatTensor(
            out_target.data.size()).fill_(source_label).cuda())

        loss_adv_target0 = loss_adv_target0 * 100
        # optimizer_ecoder.zero_grad()
        # loss_adv_target0.backward(retain_graph=True)


        # balanced_loss = loss_adv_target0 + clu_loss + 10.0*BCE_mr+torch.mul(KLD_mr, kldlamda)+10.0*BCE_ct+torch.mul(KLD_ct, kldlamda)+torch.mul(distance_loss, dislamda)+predlamda*(segloss_output+fusionsegloss_output) + \
        # 10.0*BCE_down2_ct + torch.mul(KLD_down2_ct, kldlamda) + 10.0*BCE_down2_mr + torch.mul(KLD_down2_mr, kldlamda) + torch.mul(distance_down2_loss, dislamdadown2) + predlamda * segdown2loss_output + \
        # 10.0*BCE_down4_ct + torch.mul(KLD_down4_ct, kldlamda) + 10.0*BCE_down4_mr + torch.mul(
        #     KLD_down4_mr, kldlamda) + torch.mul(distance_down4_loss, dislamdadown4) + predlamda * segdown4loss_output


        balanced_loss = loss_adv_target0 + clu_loss + 10.0*BCE_mr+torch.mul(KLD_mr, kldlamda)+10.0*BCE_ct+torch.mul(KLD_ct, kldlamda)+torch.mul(distance_loss, dislamda)+predlamda*(segloss_output+fusionsegloss_output) + \
            10.0*BCE_down2_ct + torch.mul(KLD_down2_ct, kldlamda) + 10.0*BCE_down2_mr + torch.mul(KLD_down2_mr, kldlamda) + torch.mul(distance_down2_loss, dislamdadown2) + predlamda * segdown2loss_output + \
            10.0*BCE_down4_ct + torch.mul(KLD_down4_ct, kldlamda) + 10.0*BCE_down4_mr + torch.mul(
                KLD_down4_mr, kldlamda) + torch.mul(distance_down4_loss, dislamdadown4) + predlamda * segdown4loss_output

        writer.add_scalar('distance_loss', torch.mul(
            distance_loss, dislamda).item(), i+epoch*max_iter)

        # 源域上监督学习分割损失不下降 ?
        # 1.增加网络复杂度
        # 2.提高分割损失的权重
        writer.add_scalar('fusion_segloss',
                          fusionsegloss_output.item(), i+epoch*max_iter)

        writer.add_scalar('fusionsegloss_output_dice',
                          fusionsegloss_output_dice.item(), i+epoch*max_iter)
        writer.add_scalar('fusionsegloss_output_ce',
                          fusionsegloss_output_ce.item(), i+epoch*max_iter)
        # writer.add_scalar('fusionsegloss_output_edge',
        #                   fusionsegloss_output_edge.item(), i+epoch*max_iter)
        writer.add_scalar('KLD_mr', KLD_mr.item(), i+epoch*max_iter)
        writer.add_scalar('KLD_ct', KLD_ct.item(), i+epoch*max_iter)

        # writer.add_scalar(
        #     'distance_loss', fusionsegloss_output.item(), i+epoch*max_iter)

        # print('loss_adv_target0 = {:.4f} balanced_loss = {:.4f}'.format(
        #     loss_adv_target0 * predlamda, balanced_loss))
        optim.zero_grad()
        balanced_loss.backward(retain_graph=True)
        # optim.step()

        # with torch.autograd.detect_anomaly():

        # fusionseg_mr, pred_mr, _, feat_mr, mu_mr, logvar_mr, preddown2_mr, _, featdown2_mr, mudown2_mr, logvardown2_mr, preddown4_mr, _, featdown4_mr, mudown4_mr, logvardown4_mr, info_pred_mr = encoder(
        # mr, gate)
        # out_target = model_D0(fusionseg_mr)
        # loss_adv_target0 = bce_loss(out_target, torch.FloatTensor(
        #     out_target.data.size()).fill_(source_label).cuda())

        # loss_adv_target0 = loss_adv_target0 * 100
        # # optimizer_ecoder.zero_grad()
        # loss_adv_target0.backward(retain_graph=True)
        # optimizer_ecoder.step()
        print('clu_loss = {:.4f}'.format(clu_loss.item()))
        # print('clu_loss_image = {:.4f}'.format(clu_loss_image.item()))

        # clu_loss = clu_loss * 1000
        # clu_loss.backward(retain_graph=True)

        # clu_loss_image = clu_loss_image * 1000
        # clu_loss_image.backward()

        writer.add_scalar('clu_loss', clu_loss.item(), i+epoch*max_iter)

        optim.step()
        print('loss_adv_target0 = {:.4f} balanced_loss = {:.4f}'.format(
            loss_adv_target0, balanced_loss))

        if i % 20 == 0:
            # print('epoch %d , %d th iter; seglr,ADA_totalloss,segloss,distance_loss1,distance_loss2: %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'\
            #       % (epoch, i,lr, balanced_loss.item(),BCE_mr.item(),KLD_mr.item(),BCE_ct.item(),KLD_ct.item(),fusionsegloss_output.item(),segloss_output.item(),segdown2loss_output.item(),segdown4loss_output.item(),distance_loss.item(),distance_down2_loss.item(),distance_down4_loss.item()))

            print('epoch:{},iter:{} lr={} ADA_totalloss={:.4} fusionsegloss_output={:.4} distance_loss={:.4}'.format(
                epoch, i, lr, balanced_loss.item(), fusionsegloss_output.item(), distance_loss.item()))

        i = i+1


def SegNet_test_mr(test_dir, mrSegNet, gate,epoch,ePOCH, save_DIR,save_IMG_DIR):
    criterion=0
    for dir in test_dir:
        labsname = glob.glob(dir + '*manual.nii*')
        labsname = sorted(labsname)
        
        total_dice = np.zeros((4,))
        total_Iou = np.zeros((4,))

        total_overlap =np.zeros((1,4, 5))
        total_surface_distance=np.zeros((1,4, 5))

        num = 0
        mrSegNet.eval()
        for i in range(len(labsname)):
            itklab = sitk.ReadImage(labsname[i])
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            imgname = labsname[i].replace('_manual.nii', '.nii')
            itkimg = sitk.ReadImage(imgname)
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            itkC0_image = sitk.ReadImage('/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA/Dataset/Patch192/C0_val/patient2_C0.nii.gz')
            C0_image = sitk.GetArrayFromImage(itkC0_image)

            

            # data = np.transpose(
            #     transform.resize(np.transpose(npimg, (1, 2, 0)), (96, 96),
            #                      order=3, mode='edge', preserve_range=True), (2, 0, 1))
            data=torch.from_numpy(np.expand_dims(npimg,axis=1)).type(dtype=torch.FloatTensor).cuda()
            data_C0=torch.from_numpy(np.expand_dims(C0_image,axis=1)).type(dtype=torch.FloatTensor).cuda()


            label=torch.from_numpy(nplab).cuda()

            truearg  = np.zeros((data.size(0),data.size(2),data.size(3)))

            for slice in range(data.size(0)):
                slice = slice + 9

                output,_,_, z, _, _ ,_,_,_,_,_,_,_,_,_,_,_= mrSegNet(data[slice:slice+1,:,:,:], gate)
                output_C0,_,_, z_C0, _, _ ,_,_,_,_,_,_,_,_,_,_,_= mrSegNet(data_C0[slice:slice+1,:,:,:], gate)

                # plt.imshow(data_C0[slice,0,:,:].cpu().detach().numpy())
                # plt.show()

                # plt.imshow(torch.argmax(output_C0[0], dim=0).cpu().detach().numpy())
                # plt.show()

                print(z.shape)
                # (1, 64, 192, 192)
                # plt.imshow(label[slice, 40:160, 40:160].cpu().detach().numpy())
                # plt.show()
                print(z_C0.shape)
                z = z[:, :, 40:160, 40:160]
                z_C0 = z_C0[:, :, 40:160, 40:160]
                z = torch.flatten(z,start_dim=2)
                z_C0 = torch.flatten(z_C0,start_dim=2)

                label_z = torch.zeros((z.shape[-1]))
                label_z_C0 = torch.zeros((z_C0.shape[-1]))
                label_z_C0 = 1-label_z_C0
                

                z = torch.cat((z[0], z_C0[0]), dim=1)
                print(z.shape)

                z = np.array(z.cpu().detach())
                z_C0 = np.array(z_C0.cpu().detach())
                
                print(z.shape)
                z = z.transpose((1, 0))
                print(z.shape)


                # (1, 64, 36864)
                print('starting T-SNE process')

                # label_df = torch.zeros((z.shape[0]))
                # label_df = label[slice, 40:160, 40:160].cpu()


                # label_df = torch.flatten(label_df)

                # label_df = torch.zeros((z.shape[0]))
                label_df = torch.cat((label_z,label_z_C0),dim=0)

                label_df = label_df.detach().numpy()
                print(label_df.shape)
                # start_time = time()
                z = TSNE(n_components=2, verbose=1 ,random_state=42).fit_transform(z)

                scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
                result = scaler.fit_transform(z)

                # 颜色设置
                color = ['#FFFAFA', '#BEBEBE', '#000080', '#87CEEB', '#006400',
                        '#00FF00', '#4682B4', '#D02090', '#8B7765', '#B03060']

                # 可视化展示
                plt.figure(figsize=(10, 10))
                plt.title('VarDA', fontsize='xx-large')
                # plt.title('VAMCEI(ours)')
                # plt.title('VAMCEI(ours)')


                # plt.xlim((-1.1, 1.1))
                # plt.ylim((-1.1, 1.1))
                # for i in range(len(result)):
                #     plt.text(result[i,0], result[i,1], str(y[i]), 
                #              color=color[y[i]], fontdict={'weight': 'bold', 'size': 9})
                plt.scatter(result[:,0], result[:,1], c=label_df, s=10)

                plt.show()




                # z_min, z_max = np.min(z, 0), np.max(z, 0)
                # z = (z - z_min) / (z_max - z_min)
                # df = pd.DataFrame(z, columns=['x', 'y'])  # 转换成df表
                # df.insert(loc=1, column='label', value=label_df)
                # # end_time = time()
                # print('Finished')

                # # 绘图
                # sns.scatterplot(x='x', y='y', hue='label', s=3, palette="Set2", data=df)
                # # self.set_plt(start_time, end_time, title)
                # # plt.savefig('2.jpg', dpi=400)
                # plt.show()

                truemax, truearg0 = torch.max(output, 1, keepdim=False)
                truearg0 = truearg0.detach().cpu().numpy()
                truearg[slice:slice+1,:,:]=truearg0
            #truearg = np.transpose(transform.resize(np

            #
            # truemax, truearg = torch.max(output, 1, keepdim=False)
            # truearg = truearg.detach().cpu().numpy()
            # truearg = np.transpose(transform.resize(np.transpose(truearg, (1, 2, 0)), (192,192), order=0,mode='edge', preserve_range=True), (2, 0, 1)).astype(np.int64)

            dice = dice_compute(truearg,label.cpu().numpy())
            Iou = IOU_compute(truearg,label.cpu().numpy())
            overlap_result, surface_distance_result = Hausdorff_compute(truearg,label.cpu().numpy(),itkimg.GetSpacing())

            total_dice = np.vstack((total_dice,dice))
            total_Iou = np.vstack((total_Iou,Iou))

            total_overlap = np.concatenate((total_overlap,overlap_result),axis=0)
            total_surface_distance = np.concatenate((total_surface_distance,surface_distance_result),axis=0)

            num+=1
            print(dice)

        if num==0:
            return
        else:
            meanDice = np.mean(total_dice[1:],axis=0)
            stdDice = np.std(total_dice[1:],axis=0)

            meanIou = np.mean(total_Iou[1:],axis=0)
            stdIou = np.std(total_Iou[1:],axis=0)

            mean_overlap = np.mean(total_overlap[1:], axis=0)
            std_overlap = np.std(total_overlap[1:], axis=0)

            mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
            std_surface_distance = np.std(total_surface_distance[1:], axis=0)

            if 'Vali' in dir:
                phase='validate'
            else:
                criterion = np.mean(meanDice[1:])
                phase='test'
            with open("%s/lge_testout_index.txt" % (save_DIR), "a") as f:
                f.writelines(["\n\nepoch:", str(epoch), " ",phase," ", "\n","meanDice:",""\
                                 ,str(meanDice.tolist()),"stdDice:","",str(stdDice.tolist()),"","\n","meanIou:","",str(meanIou.tolist()),"stdIou:","",str(stdIou.tolist()), \
                                  "", "\n\n","jaccard, dice, volume_similarity, false_negative, false_positive:", "\n","mean:", str(mean_overlap.tolist()),"\n", "std:", "", str(std_overlap.tolist()), \
                                  "", "\n\n","hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance:", "\n","mean:", str(mean_surface_distance.tolist()), "\n","std:", str(std_surface_distance.tolist())])
    return criterion


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    cudnn.benchmark = True

    PredLamda = 3e3
    DisLamda = 2e-3
    DisLamdaDown2 = 2e-3
    DisLamdaDown4 = 2e-4
    InfoLamda = 0.0

    sample_nums = [45]
    writer = SummaryWriter(
        '/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA/code/original/log')

    for sample_num in sample_nums:

        SAVE_DIR = prefix+'/save_train_param'+'_num'+str(sample_num)
        SAVE_IMG_DIR = prefix+'/save_test_label'+'_num'+str(sample_num)
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(SAVE_IMG_DIR):
            os.mkdir(SAVE_IMG_DIR)

        print(torch.cuda.is_available())
        vaeencoder = VAE()
        vaeencoder = vaeencoder.cuda()

        source_vaedecoder = VAEDecode()
        source_vaedecoder = source_vaedecoder.cuda()

        source_down2_vaedecoder = VAEDecode_down2()
        source_down2_vaedecoder = source_down2_vaedecoder.cuda()

        source_down4_vaedecoder = VAEDecode_down4()
        source_down4_vaedecoder = source_down4_vaedecoder.cuda()

        target_vaedecoder = VAEDecode()
        target_vaedecoder = target_vaedecoder.cuda()

        target_down2_vaedecoder = VAEDecode_down2()
        target_down2_vaedecoder = target_down2_vaedecoder.cuda()

        target_down4_vaedecoder = VAEDecode_down4()
        target_down4_vaedecoder = target_down4_vaedecoder.cuda()

        # 定义鉴别器
        model_D0 = FCDiscriminator(num_classes=4).cuda()
        model_D0.train()

        # 定义鉴别器的优化器
        optimizer_D0 = optim.Adam(model_D0.parameters(
        ), lr=LR_D, betas=(0.9, 0.99))
        optimizer_ecoder = optim.Adam(vaeencoder.parameters(
        ), lr=LR_G, weight_decay=WEIGHT_DECAY)
        optimizer_ecoder.zero_grad()

        optimizer_D0.zero_grad()

        #Infonet = InfoNet().cuda()

        # DistanceNet = Gaussian_Distance_KL_Jensen(KERNEL)  # 64,Num_Feature2,(12,12)

        DistanceNet = Gaussian_Distance_KL_Jensen(KERNEL)
        DistanceNet = DistanceNet.cuda()
        # DistanceNet2 = nn.DataParallel(DistanceNet2, device_ids=[0,1])

        DA_optim = torch.optim.Adam([{'params': vaeencoder.parameters()},
                                     {'params': source_vaedecoder.parameters()},
                                     {'params': source_down2_vaedecoder.parameters()},
                                     {'params': source_down4_vaedecoder.parameters()},
                                     {'params': target_vaedecoder.parameters()},
                                     {'params': target_down2_vaedecoder.parameters()},
                                     {'params': target_down4_vaedecoder.parameters()}], lr=LR,
                                    weight_decay=WEIGHT_DECAY)

        SourceData = C0_TrainSet(dataset_dir, 35, times=4)
        SourceData_loader = DataLoader(
            SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True, drop_last=True)

        # SourceData_val = C0_ValSet(dataset_dir, 5)
        # SourceData_val_loader = DataLoader(SourceData_val, batch_size=1, shuffle=False, num_workers=WORKERSNUM,
        #                                pin_memory=True, drop_last=True)

        # sample_num = 45
        TargetData = LGE_TrainSet(dataset_dir, sample_num, times=4)
        TargetData_loader = DataLoader(
            TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True, drop_last=True)

        

        # cycleGAN后的真假LGE图像对齐
        # SourceData = C0_TrainSet(dataset_dir, 35, times=4)
        # SourceData_loader = DataLoader(
        #     SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True, drop_last=True)
        # TargetData = LGE_TrainSet(dataset_dir, sample_num, times=4)
        # TargetData_loader = DataLoader(
        #     TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True, drop_last=True)
        # dir_img = Path(
        #     '/home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/fake_lge_2')
        # dir_mask = Path(
        #     '/home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/c0gt')

        # original_img = Path(
        #     # '/home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/original_img'
        #     '/home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_nii45_for_challenge19/lge'
        # )

        # train_set_fake = BasicDataset(
        #     dir_img, dir_mask, multiple = 4)
        # train_set_original = BasicDatasetOriginal(
        #     original_img, multiple = 4)
            
        # loader_args = dict(batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True, drop_last=True)

        # SourceData_loader = DataLoader(train_set_fake, **loader_args)
        # TargetData_loader = DataLoader(
        #     train_set_original, **loader_args)

        vaeencoder.apply(init_weights)
        source_vaedecoder.apply(init_weights)
        source_down2_vaedecoder.apply(init_weights)
        source_down4_vaedecoder.apply(init_weights)
        target_vaedecoder.apply(init_weights)
        target_down2_vaedecoder.apply(init_weights)
        target_down4_vaedecoder.apply(init_weights)
        model_D0.apply(init_weights)

        criterion = 0
        best_epoch = 0

        # q_source = Queue(maxsize=10)
        # q_target = Queue(maxsize=10)

        # pro_global_source = prototype_g(num_class=4, feat_dim=64)

        


        for epoch in range(EPOCH):

            # criter = SegNet_test_mr(
            #     TestDir, vaeencoder, 0, epoch, EPOCH, SAVE_DIR, SAVE_IMG_DIR)
                
            vaeencoder.train()
            source_vaedecoder.train()
            source_down2_vaedecoder.train()
            source_down4_vaedecoder.train()
            target_vaedecoder.train()
            target_down2_vaedecoder.train()
            target_down4_vaedecoder.train()

            model_D0.eval()

            # 生成器优化
            ADA_Train(SourceData_loader, TargetData_loader, vaeencoder, source_vaedecoder, source_down2_vaedecoder, source_down4_vaedecoder, target_vaedecoder,
                      target_down2_vaedecoder, target_down4_vaedecoder, 1.0, DistanceNet, LR, KLDLamda, PredLamda, DisLamda, DisLamdaDown2, DisLamdaDown4, epoch, DA_optim, SAVE_DIR, model_D0, mode, writer)

            vaeencoder.eval()
            model_D0.train()

            # 鉴别器优化
            A_iter = iter(SourceData_loader)
            B_iter = iter(TargetData_loader)
            i = 0
            if mode == 'Vanilla':
                bce_loss = torch.nn.BCEWithLogitsLoss()
            elif mode == 'LS':
                bce_loss = torch.nn.MSELoss()
            while i < len(A_iter) and i < len(B_iter):

                i += 1

                optimizer_D0.zero_grad()
                ct, ct_down2, ct_down4, label, label_down2, label_down4, _ = A_iter.next()
                mr, mr_down2, mr_down4, _ = B_iter.next()

                ct = ct.cuda()
                mr = mr.cuda()
                fusionseg, predict, out_ct, feat_ct, mu_ct, logvar_ct, _, outdown2_ct, featdown2_ct, mudown2_ct, logvardown2_ct, _, outdown4_ct, featdown4_ct, mudown4_ct, logvardown4_ct, info_pred_ct = vaeencoder(
                    ct, 1.0)
                fusionseg_mr, predict_mr, out_ct, feat_ct, mu_ct, logvar_ct, _, outdown2_ct, featdown2_ct, mudown2_ct, logvardown2_ct, _, outdown4_ct, featdown4_ct, mudown4_ct, logvardown4_ct, info_pred_ct = vaeencoder(
                    mr, 1.0)

                out_source = model_D0(fusionseg)
                out_target = model_D0(fusionseg_mr)

                loss_adv_target = bce_loss(out_target, torch.FloatTensor(
                    out_target.data.size()).fill_(target_label).cuda())

                loss_adv_source = bce_loss(out_source, torch.FloatTensor(
                    out_source.data.size()).fill_(source_label).cuda())

                # loss = loss_adv_target+loss_adv_source

                loss_adv_target.backward()
                loss_adv_source.backward()

                optimizer_D0.step()
                print('loss_adv_target = {:.4f} loss_adv_source = {:.4f}'.format(
                    loss_adv_target, loss_adv_source))

            criter = SegNet_test_mr(
                TestDir, vaeencoder, 0, epoch, EPOCH, SAVE_DIR, SAVE_IMG_DIR)

            if criter > criterion:
                best_epoch = epoch
                criterion = criter
                print('average dice = {}'.format(criter))
                print("Save model!")
                torch.save(vaeencoder.state_dict(), os.path.join(
                    SAVE_DIR, 'encoder_param.pkl'))
                torch.save(source_vaedecoder.state_dict(),
                           os.path.join(SAVE_DIR, 'decoderA_param.pkl'))
                torch.save(source_down2_vaedecoder.state_dict(),
                           os.path.join(SAVE_DIR, 'decoderAdown2_param.pkl'))
                torch.save(source_down4_vaedecoder.state_dict(),
                           os.path.join(SAVE_DIR, 'decoderAdown4_param.pkl'))
                torch.save(target_vaedecoder.state_dict(),
                           os.path.join(SAVE_DIR, 'decoderB_param.pkl'))
                torch.save(target_down2_vaedecoder.state_dict(),
                           os.path.join(SAVE_DIR, 'decoderBdown2_param.pkl'))
                torch.save(target_down4_vaedecoder.state_dict(),
                           os.path.join(SAVE_DIR, 'decoderBdown4_param.pkl'))
        print('\n')
        print('\n')
        print('best epoch:%d' % (best_epoch))
        with open("%s/lge_testout_index.txt" % (SAVE_DIR), "a") as f:
            f.writelines(["\n\nbest epoch:%d" % (best_epoch)])

        del vaeencoder, source_vaedecoder, source_down2_vaedecoder, source_down4_vaedecoder, target_vaedecoder, target_down2_vaedecoder, target_down4_vaedecoder


def test():
    vaeencoder = VAE().cuda()
    vaeencoder.load_state_dict(torch.load(
        '/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA/base/encoder_param.pkl'))
    # dataset_dir = './Dataset/Patch192'
    # TestDir = [dataset_dir+'/LGE/LGE_Test/',dataset_dir+'/LGE/LGE_Vali/']
    TestDir = ['/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA/Dataset/Patch192/LGE_test/']
    # TestDir = ['/home/hfcui/cmrseg2019_project/cmr2019_data/C0LET2_gt_for_challenge19/C0LET2_gt_for_challenge19/lge/']
    epoch = 1
    EPOCH = 30
    sample_num = 45
    prefix = '.'
    SAVE_DIR = prefix + '/save_train_param' + '_num' + str(sample_num)
    SAVE_IMG_DIR = prefix+'/save_test_label'+'_num'+str(sample_num)
    SAVE_DIR = '/media/hfcui/89b4f7a9-e28f-4fee-aef5-bc28c6f65775/VarDA/save_train_param_num45'
    criter = SegNet_test_mr(TestDir, vaeencoder, 0, epoch,
                            EPOCH, SAVE_DIR, SAVE_IMG_DIR)
    print(criter)
    return


if __name__ == '__main__':
    test()
