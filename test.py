from sklearn.manifold import TSNE
from time import time
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class T_sne_visual():
    def __init__(self, model, dataset, dataloader):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.class_list=dataset.classes
    def visual_dataset(self):
        imgs = []
        labels = []
        for img, label in self.dataset:
            imgs.append(np.array(img).transpose((2, 1, 0)).reshape(-1))
            tag = self.class_list[label]
            labels.append(tag)
        self.t_sne(np.array(imgs), labels,title=f'Dataset visualize result\n')

    def visual_feature_map(self, layer):
        self.model.eval()
        with torch.no_grad():
            self.feature_map_list = []
            labels = []
            getattr(self.model, layer).register_forward_hook(self.forward_hook)
            for img, label in self.dataloader:
                img=img.cuda()
                self.model(img, 1.0)

                label = label.reshape((label.shape[0]*label.shape[1]*label.shape[2]))
                # print(label.shape)

                labels.append(label)
                # for i in label.tolist():
                #     tag=self.class_list[i]
                #     labels.append(tag)
            self.feature_map_list = torch.cat(self.feature_map_list,dim=0)
            
            # print(self.feature_map_list.shape)

            labels = torch.cat(labels,dim=0)
            
            self.feature_map_list=torch.flatten(self.feature_map_list,start_dim=1)
            print(self.feature_map_list.shape)
            print(labels.shape)
            # self.feature_map_list = torch.where((labels!=0),self.feature_map_list,None)
            # labels = torch.where((labels!=0),labels,None)
            self.feature_map_list = np.array(self.feature_map_list.cpu())
            labels = np.array(labels)
            num = labels.shape
            for i in range(num[-1]):
                # print(labels[num[-1]-i-1])
                # print(num[-1]-i-1)
                if labels[num[-1]-i-1] == 0:
                    # print('delete...')
                    labels = np.delete(labels,num[-1]-i-1)
                    self.feature_map_list = np.delete(self.feature_map_list,num[-1]-i-1,axis=0)
            print(self.feature_map_list.shape)
            print(labels.shape)
                    
            
            
            self.t_sne(self.feature_map_list, labels ,title=f'{layer} resnet feature map\n')

    def forward_hook(self, model, input, output):
        # print(output.shape)
        output = output.reshape((output.shape[0]*output.shape[-2]*output.shape[-1],-1))
        self.feature_map_list.append(output)

    def set_plt(self, start_time, end_time,title):
        plt.title(f'{title} time consume:{end_time - start_time:.3f} s')
        plt.legend(title='')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

    def t_sne(self, data, label,title):
        # t-sne处理
        # print(data.shape)
        print('starting T-SNE process')
        # start_time = time()
        data = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
        df.insert(loc=1, column='label', value=label)
        # end_time = time()
        print('Finished')

        # 绘图
        sns.scatterplot(x='x', y='y', hue='label', s=3, palette="Set2", data=df)
        # self.set_plt(start_time, end_time, title)
        plt.savefig('2.jpg', dpi=400)
        plt.show()

from utils_for_transfer import *
dataset_dir = '/home/hfcui/cmrseg2019_project/VarDA/Dataset/Patch192'

# trans=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# imgset=torchvision.datasets.ImageFolder(root=r'C:\Users\Administrator\Desktop\imageset',transform=trans)
# img_loader=DataLoader(imgset,batch_size=16,shuffle=True)
# net=torchvision.models.resnet34(pretrained=False).cuda()

TargetData = C0_TrainSet(dataset_dir, 1, times=1)
TargetData_loader = DataLoader(
            TargetData, batch_size=1, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)

net = VAE().cuda()
net.load_state_dict(torch.load(
    '/home/hfcui/cmrseg2019_project/VarDA/save_train_param_num45/encoder_param.pkl'))

t = T_sne_visual(net, TargetData, TargetData_loader)
# t.visual_dataset()
t.visual_feature_map('fc1')