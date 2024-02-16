
import numpy as np
import torch
from torch import nn



def edge_loss(out, target, cuda=True):
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    #  torch.nn.functional.conv2d
    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3 , stride=1, padding=1, bias=False)
    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

    if cuda:
        weights_x = weights_x.cuda()
        weights_y = weights_y.cuda()

    convx.weight = nn.Parameter(weights_x)
    convx.requires_grad_(False)
    convy.weight = nn.Parameter(weights_y)
    convy.requires_grad_(False)
    
    g1_x = convx(out)
    g2_x = convx(target)
    g1_y = convy(out)
    g2_y = convy(target)

    g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
    g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

    return torch.mean((g_1 - g_2).pow(2))