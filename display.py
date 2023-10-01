import os
import torch
from utils_for_transfer import *
from Adversarial_DA_seg_trainer import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 7.0)
# vaeencoder = VAE().cuda()
# vaeencoder.load_state_dict(torch.load(
#         '/home/hfcui/cmrseg2019_project/VarDA/save_train_param_num45/encoder_param.pkl'))
def id2trainId(label):
    # left ventricular (LV) blood pool (labelled 500),
    # right ventricular blood pool (600),
    # LV normal myocardium (200),
    # LV myocardial edema (1220),
    # LV myocardial scars (2221),
    shape = label.shape
    results_map = np.zeros((4, shape[0], shape[1]))

    LV = (label == 1)
    RV = (label == 2)
    MY = (label == 3)
    # edema = (label == 4)
    # scars = (label == 5)

    background = np.logical_not(LV + RV + MY)

    results_map[0, :, :] = np.where(background, 1, 0)
    results_map[1, :, :] = np.where(LV, 1, 0)
    results_map[2, :, :] = np.where(RV, 1, 0)
    results_map[3, :, :] = np.where(MY, 1, 0)
    # results_map[4, :, :] = np.where(edema, 1, 0)
    # results_map[5, :, :] = np.where(scars, 1, 0)
    return results_map

display_list = [
    {
        'ex_name':'Baseline',
        'model_path':'/home/hfcui/cmrseg2019_project/VarDA/base/encoder_param.pkl'
    },
    {
        'ex_name':'Baseline_Pro',
        'model_path':'/home/hfcui/cmrseg2019_project/VarDA/base_pro/encoder_param.pkl'
    },
    {
        'ex_name':'Baseline_L_dis',
        'model_path':'/home/hfcui/cmrseg2019_project/VarDA/base_dis/encoder_param.pkl'
    },
    {
        'ex_name':'Baseline_L_dis_Pro',
        'model_path':'/home/hfcui/cmrseg2019_project/VarDA/base_pro_dis/encoder_param.pkl'
    },
    {
        'ex_name':'Baseline_L_dis_Pro_D(ours)',
        'model_path':'/home/hfcui/cmrseg2019_project/VarDA/base_D_pro_dis/encoder_param.pkl'
    },
]

model_list = []
name_list = []
TestDir = ['/home/hfcui/cmrseg2019_project/VarDA/Dataset/Patch192/LGE_1/']
for i in range(len(display_list)):
    ex_name = display_list[i]['ex_name']
    model_path = display_list[i]['model_path']
    vaeencoder = VAE().cuda()
    vaeencoder.load_state_dict(torch.load(model_path))
    vaeencoder.eval()
    model_list.append(vaeencoder)
    name_list.append(ex_name)

criterion = 0
save_path = '/home/hfcui/cmrseg2019_project/VarDA/code/original/test_imgs'
colors1 = ['c']
colors2 = ['tomato']
colors3 = ['wheat']
import matplotlib

cmap1 = matplotlib.colors.ListedColormap(colors1)
cmap2 = matplotlib.colors.ListedColormap(colors2)
cmap3 = matplotlib.colors.ListedColormap(colors3)

for dir in TestDir:
    labsname = glob.glob(dir + '*manual.nii*')

    total_dice = np.zeros((4,))
    total_Iou = np.zeros((4,))

    total_overlap = np.zeros((1, 4, 5))
    total_surface_distance = np.zeros((1, 4, 5))

    num = 0
    # mrSegNet.eval()
    for i in range(len(labsname)):
        itklab = sitk.ReadImage(labsname[i])
        nplab = sitk.GetArrayFromImage(itklab)
        nplab = (nplab == 500) * 1 + (nplab == 600) * \
            2 + (nplab == 200) * 3

        imgname = labsname[i].replace('_manual.nii', '.nii')
        itkimg = sitk.ReadImage(imgname)
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
        npimg = npimg.astype(np.float32)

        # data = np.transpose(
        #     transform.resize(np.transpose(npimg, (1, 2, 0)), (96, 96),
        #                      order=3, mode='edge', preserve_range=True), (2, 0, 1))
        data = torch.from_numpy(np.expand_dims(npimg, axis=1)).type(
            dtype=torch.FloatTensor).cuda()

        label = torch.from_numpy(nplab).cuda()

        

        truearg = np.zeros((len(model_list),data.size(0), data.size(2), data.size(3)))

        # data (deep, 1, h, w)
        # truearg (model_num, deep, h, w)
        # fig, ax = plt.subplots(5, len(model_list)+1)
        for slice in range(data.size(0)):
            print(data.shape)
            plt.imshow(data[slice, 0, :, :][16:176, 16:176].detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            # plt.show()

            plt.savefig(os.path.join(save_path, 'image_'+str(slice)+'.jpg'),bbox_inches='tight', pad_inches=0)
            

            
            print(label.shape)
            plt.imshow(label[slice, :, :][16:176, 16:176].detach().cpu().numpy())
            plt.axis('off')
            # plt.show()
            plt.savefig(os.path.join(save_path, 'label_'+str(slice)+'.jpg'),bbox_inches='tight', pad_inches=0)

            # plt.savefig(os.path.join(save_path, 'image_'+str(slice)+'.jpg'),bbox_inches='tight', pad_inches=0)
            
            for j in range(len(model_list)):
                print(j)
                mrSegNet = model_list[j]
                name = name_list[j]
                output, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = mrSegNet(
                    data[slice:slice+1, :, :, :], 1.0)

                truemax, truearg0 = torch.max(output, 1, keepdim=False)
                truearg0 = truearg0.detach().cpu().numpy()

                truearg0 = np.where((truearg0 == 1), 50, truearg0)
                truearg0 = np.where((truearg0 == 2), 100, truearg0)
                truearg0 = np.where((truearg0 == 3), 150, truearg0)

                truearg0 = np.where((truearg0 == 50), 3, truearg0)
                truearg0 = np.where((truearg0 == 100), 1, truearg0)
                truearg0 = np.where((truearg0 == 150), 2, truearg0)

                truearg[j, slice:slice+1, :, :] = truearg0
                print(name)

                plt.imshow(truearg[j, slice, :, :][16:176, 16:176])
                plt.axis('off')
                # plt.show()

                plt.savefig(os.path.join(save_path, name+'_'+str(slice)+'.jpg'),bbox_inches='tight', pad_inches=0)


                # pre = id2trainId(truearg0[0])
        #         if data.size(0) >= 14:
        #             if slice % 3 == 0 and slice < 15:
        #                 if slice == 0:
        #                     ax[slice//3, j].set_title(name_list[j])
        #                 ax[slice//3, j].imshow(data[slice, 0].cpu().detach().numpy(), 'gray')

        #                 pre_1 = np.ma.masked_where(pre[1] == 0, pre[0])
        #                 ax[slice//3, j].imshow(pre_1, cmap=cmap1, alpha=0.7)

        #                 pre_2 = np.ma.masked_where(pre[2] == 0, pre[0])
        #                 ax[slice//3, j].imshow(pre_2, cmap=cmap2, alpha=0.7)

        #                 pre_3 = np.ma.masked_where(pre[3] == 0, pre[0])
        #                 ax[slice//3, j].imshow(pre_3, cmap=cmap3, alpha=0.7)
        #                 ax[slice//3, j].axis('off')
            
        #     label_ = id2trainId(label[slice].cpu().detach().numpy())
        #     if data.size(0) >= 14:
        #         if slice % 3 == 0 and slice < 15:
        #             if slice == 0:
        #                 ax[slice//3, len(model_list)].set_title('Ground truth')
        #             ax[slice//3, len(model_list)].imshow(data[slice, 0].cpu().detach().numpy(), 'gray')

        #             pre_1 = np.ma.masked_where(label_[1] == 0, label_[0])
        #             ax[slice//3, len(model_list)].imshow(pre_1, cmap=cmap1, alpha=0.7)

        #             pre_2 = np.ma.masked_where(label_[2] == 0, label_[0])
        #             ax[slice//3, len(model_list)].imshow(pre_2, cmap=cmap2, alpha=0.7)

        #             pre_3 = np.ma.masked_where(label_[3] == 0, label_[0])
        #             ax[slice//3, len(model_list)].imshow(pre_3, cmap=cmap3, alpha=0.7)
        #             ax[slice//3, len(model_list)].axis('off')
        
        # # plt.rcParams['figure.figsize'] = (16.0, 8.0)
        # plt.tight_layout(pad=0.5)
        # # plt.show()
        # plt.savefig(os.path.join('/home/hfcui/cmrseg2019_project/VarDA/code/original_end2end/result', os.path.basename(labsname[i]).split('_')[0]+'.jpg'))


                

        # fig, ax = plt.subplots(5, len(model_list)+1)
