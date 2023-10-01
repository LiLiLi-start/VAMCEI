""" Full assembly of the parts to form the complete network """

from .unet_parts import *
def conv2d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class conv2d_same_pad(nn.Conv2d):
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


class IB3(nn.Module):
    def __init__(self, in_channels, flt=64):
        super(IB3, self).__init__()
        self.flt = flt
        self.conv1 = nn.Sequential(
            conv2d_same_pad(in_channels, self.flt, kernel_size=1, bias=True),
        )
        self.conv3 = nn.Sequential(
            conv2d_same_pad(in_channels, self.flt, kernel_size=3, bias=True),
        )
        self.conv5 = nn.Sequential(
            conv2d_same_pad(in_channels, self.flt, kernel_size=5, bias=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(self.flt*3, self.flt, kernel_size=1, bias=True),
        )

    def forward(self, input):
        conv1=self.conv1(input)
        conv3=self.conv3(input)
        conv5=self.conv5(input)
        concate=torch.cat((conv1,conv3,conv5),1)
        return self.conv(concate)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.IB3=IB3(n_channels,32)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x=self.IB3(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
