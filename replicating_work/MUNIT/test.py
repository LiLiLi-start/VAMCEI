"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from utils import get_data_loader_folder
import matplotlib.pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='/home/hfcui/MUNIT/configs/mmwhs.yaml', help="net configuration")
parser.add_argument('--input', type=str, default='/home/hfcui/MUNIT/mmwhs_root/testA', help="input image path")
parser.add_argument('--output_folder', type=str, default='/home/hfcui/MUNIT/outputs/myops/test', help="output image path")
parser.add_argument('--checkpoint', type=str, default='/home/hfcui/MUNIT/outputs/mmwhs/checkpoints/gen_00460000.pt', help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=5, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

batch_size = 1
new_size_a = 256
num_workers = 1
height = width = 192

test_loader_a = get_data_loader_folder(os.path.join(config['data_root'], 'testA'), batch_size, False,new_size_a, height, width, num_workers, True, isTest=True)
item = 0
with torch.no_grad():

    # 定义数据处理 读取数据
    for image, save_path in test_loader_a:
        # print(save_path)
        deep = image.shape[1]
        generate_lge = np.zeros((deep, 192, 192))
        style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
        style = style_rand
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            
            for i in range(deep):
                image_i = image[:, i:i+1]

                item = item+1
                image_i = image_i.cuda()

                style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

                # Start testing
                content, _ = encode(image_i)
                outputs = decode(content, s)
                generate_lge[i] = outputs[0, 0].cpu().detach().numpy()
                # if i == 0:
                
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(image_i[0, 0].cpu().detach().numpy(), cmap='gray')
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(outputs[0, 0].cpu().detach().numpy(), cmap='gray')
                #     plt.show()
                
            # generate_lge
            np.save(save_path[0]+str(j)+'.npy', generate_lge)
                    
                    
                    # outputs = (outputs + 1) / 2.
                    # path = os.path.join(opts.output_folder, 'output{:03d}-{:03d}.jpg'.format(item, j))
                    # vutils.save_image(outputs.data, path, padding=0, normalize=True)


