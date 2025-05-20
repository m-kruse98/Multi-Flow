'''This Code is based on the FrEIA Framework, source: https://github.com/VLL-HD/FrEIA
It is a assembly of the necessary modules/functions from FrEIA that are needed for our purposes.'''
import torch
import torch.nn as nn
from math import exp
import numpy as np

import warnings
VERBOSE = False


class NaiveCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(NaiveCrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = 'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.up_conv10 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv32 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv43 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)

    
        self.down_conv01 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.down_conv12 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.down_conv23 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.down_conv34 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)

        self.lr = nn.ReLU() # nn.LeakyReLU(self.leaky_slope)

    def forward(self, x0, x1, x2, x3, x4):
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)

        y1_up = self.up_conv10(y1)
        y2_up = self.up_conv21(y2)
        y3_up = self.up_conv32(y3)
        y4_up = self.up_conv43(y4)

        y0_down = self.down_conv01(y0)
        y1_down = self.down_conv12(y1)
        y2_down = self.down_conv23(y2)
        y3_down = self.down_conv34(y3)

        out0 = out0 + y1_up
        out1 = out1 + y0_down + y2_up
        out2 = out2 + y1_down + y3_up
        out3 = out2 + y2_down + y4_up
        out4 = out2 + y3_down

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4
    
    
    
# connections just as in CS-Flow, but neighboring views are connected. all side-views connect to the/ neighbor the top view 
class NeighboringCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(NeighboringCrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate" #  'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cross_conv12 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv14 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv23 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv32 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv34 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv41 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv43 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        
        
        self.topview_up0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up2 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up3 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        
        self.topview_down0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down2 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down3 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        
        
        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2, x3, x4):
        # x0 is top view, others are side views         
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)

        top1_up = self.topview_up0(y0)
        top2_up = self.topview_up1(y0)
        top3_up = self.topview_up2(y0)
        top4_up = self.topview_up3(y0)
        
        top1_down = self.topview_down0(y1)
        top2_down = self.topview_down1(y2)
        top3_down = self.topview_down2(y3)
        top4_down = self.topview_down3(y4)


        y21 = self.cross_conv21(y2)
        y41 = self.cross_conv41(y4)
        
        y12 = self.cross_conv12(y1)
        y32 = self.cross_conv32(y3)
        
        y23 = self.cross_conv23(y2)
        y43 = self.cross_conv43(y4)
        
        y14 = self.cross_conv14(y1)
        y34 = self.cross_conv34(y3)
    
        
        out0 = out0 + 0.25 * (top1_down + top2_down + top3_down + top4_down)
        out1 = out1 + (1 / 3) * (top1_up + y21 + y41)
        out2 = out2 + (1 / 3) * (top2_up + y12 + y32)
        out3 = out3 + (1 / 3) * (top3_up + y23 + y43)
        out4 = out4 + (1 / 3) * (top4_up + y14 + y34)
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, channels_out, kernel_size, pad, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.conv = nn.Conv2d(embed_dim, channels_out,  #
                            kernel_size=kernel_size, padding=pad,
                            padding_mode="replicate", dilation=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()
        
    def forward(self, x, k):
        B, N, H, W = x.shape
        
        x = x.view((x.shape[0], x.shape[1], -1)).permute(0, 2, 1)
        k = k.view((k.shape[0], k.shape[1], -1)).permute(0, 2, 1)
        
        out, _ = self.att(k, k, k)
        out = self.norm(self.act(out))
        out = out.permute(0, 2, 1).view((B, N, H, W))
        out = self.conv(out)
        return out
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, channels_out, kernel_size, pad, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.conv = nn.Conv2d(embed_dim, channels_out,  #
                            kernel_size=kernel_size, padding=pad,
                            padding_mode="replicate", dilation=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()
        
    def forward(self, x, k):
        B, N, H, W = x.shape
        x = x.view((x.shape[0], x.shape[1], -1)).permute(0, 2, 1)
        k = k.view((k.shape[0], k.shape[1], -1)).permute(0, 2, 1)
        
        out, _ = self.att(x, k, k)
        out = self.norm(self.act(out))
        out = out.permute(0, 2, 1).view((B, N, H, W))
        out = self.conv(out)
        return out

# connections just as in CS-Flow, but neighboring views are connected. all side-views connect to the/ neighbor the top view 
class NeighboringCrossAttention(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(NeighboringCrossAttention, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate"

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cross_conv12 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv14 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv21 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv23 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv32 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv34 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv41 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv43 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        
        self.topview_up0 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_up1 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_up2 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_up3 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        
        self.topview_down0 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_down1 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_down2 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_down3 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        
        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2, x3, x4):
        # x0 is top view, others are side views         
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)
        
        top1_up = self.topview_up0(y1, y0)
        top2_up = self.topview_up1(y2, y0)
        top3_up = self.topview_up2(y3, y0)
        top4_up = self.topview_up3(y4, y0)
        
        top1_down = self.topview_down0(y0, y1)
        top2_down = self.topview_down1(y0, y2)
        top3_down = self.topview_down2(y0, y3)
        top4_down = self.topview_down3(y0, y4)


        y21 = self.cross_conv21(y1, y2)
        y41 = self.cross_conv41(y1, y4)
        
        y12 = self.cross_conv12(y2, y1)
        y32 = self.cross_conv32(y2, y3)
        
        y23 = self.cross_conv23(y3, y2)
        y43 = self.cross_conv43(y3, y4)
        
        y14 = self.cross_conv14(y4, y1)
        y34 = self.cross_conv34(y4, y3)
        
        out0 = out0 + 0.25 * (top1_down + top2_down + top3_down + top4_down)
        out1 = out1 + (1 / 3) * (top1_up + y21 + y41)
        out2 = out2 + (1 / 3) * (top2_up + y12 + y32)
        out3 = out3 + (1 / 3) * (top3_up + y23 + y43)
        out4 = out4 + (1 / 3) * (top4_up + y14 + y34)
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4
    
# connections just as in CS-Flow, but neighboring views are connected. all side-views connect to the/ neighbor the top view 
class NeighboringSelfAttention(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(NeighboringSelfAttention, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate" #  'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
     
        
        self.cross_conv12 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv14 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv21 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv23 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv32 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv34 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv41 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv43 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        
        self.topview_up0 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_up1 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_up2 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_up3 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        
        self.topview_down0 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_down1 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_down2 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_down3 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        
        
        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2, x3, x4):
        # x0 is top view, others are side views         
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)
        
        top1_up = self.topview_up0(y1, y0)
        top2_up = self.topview_up1(y2, y0)
        top3_up = self.topview_up2(y3, y0)
        top4_up = self.topview_up3(y4, y0)
        
        top1_down = self.topview_down0(y0, y1)
        top2_down = self.topview_down1(y0, y2)
        top3_down = self.topview_down2(y0, y3)
        top4_down = self.topview_down3(y0, y4)


        y21 = self.cross_conv21(y1, y2)
        y41 = self.cross_conv41(y1, y4)
        
        y12 = self.cross_conv12(y2, y1)
        y32 = self.cross_conv32(y2, y3)
        
        y23 = self.cross_conv23(y3, y2)
        y43 = self.cross_conv43(y3, y4)
        
        y14 = self.cross_conv14(y4, y1)
        y34 = self.cross_conv34(y4, y3)
    
        
        out0 = out0 + 0.25 * (top1_down + top2_down + top3_down + top4_down)
        out1 = out1 + (1 / 3) * (top1_up + y21 + y41)
        out2 = out2 + (1 / 3) * (top2_up + y12 + y32)
        out3 = out3 + (1 / 3) * (top3_up + y23 + y43)
        out4 = out4 + (1 / 3) * (top4_up + y14 + y34)
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4
    
    
# only connect all side views with each other, without the top view
class NeighboringOnlyCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(NeighboringOnlyCrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate" #  'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cross_conv12 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv14 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv23 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv32 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv34 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv41 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv43 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2, x3, x4):
        # x0 is top view, others are side views         
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)


        y21 = self.cross_conv21(y2)
        y41 = self.cross_conv41(y4)
        
        y12 = self.cross_conv12(y1)
        y32 = self.cross_conv32(y3)
        
        y23 = self.cross_conv23(y2)
        y43 = self.cross_conv43(y4)
        
        y14 = self.cross_conv14(y1)
        y34 = self.cross_conv34(y3)
    
        
        out0 = out0 # + 0.25 * (top1_down + top2_down + top3_down + top4_down)
        out1 = out1 + (1 / 2) * (y21 + y41)
        out2 = out2 + (1 / 2) * (y12 + y32)
        out3 = out3 + (1 / 2) * (y23 + y43)
        out4 = out4 + (1 / 2) * (y14 + y34)
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4
    
# only connect all side views to the top view
class TopNeighboringCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(TopNeighboringCrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate" #  'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.topview_up0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up2 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up3 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        
        self.topview_down0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down2 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down3 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        
        
        self.lr = nn.ReLU() # nn.LeakyReLU(self.leaky_slope)

    def forward(self, x0, x1, x2, x3, x4):
        # x0 is top view, others are side views         
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)

        top1_up = self.topview_up0(y0)
        top2_up = self.topview_up1(y0)
        top3_up = self.topview_up2(y0)
        top4_up = self.topview_up3(y0)
        
        top1_down = self.topview_down0(y1)
        top2_down = self.topview_down1(y2)
        top3_down = self.topview_down2(y3)
        top4_down = self.topview_down3(y4)

    
        
        out0 = out0 + 0.25 * (top1_down + top2_down + top3_down + top4_down)
        out1 = out1 + top1_up
        out2 = out2 + top2_up
        out3 = out3 + top3_up
        out4 = out4 + top4_up
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4
    
    
    
    
# randomly connect neighboring views
class RandomNeighboringCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(RandomNeighboringCrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate" #  'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cross_conv12 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv14 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv23 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv32 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv34 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv41 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv43 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        
        
        
        
        self.topview_up0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up2 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up3 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        
        self.topview_down0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down2 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down3 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        
        perm = np.random.permutation(5)
        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i

        self.perm = torch.LongTensor(perm)
        self.perm_inv = torch.LongTensor(perm_inv)
        
        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2, x3, x4):
        
        
        to_permute = [x0,x1,x2,x3,x4]
        x0,x1,x2,x3,x4 = [to_permute[self.perm[i]] for i in range(5)]
        
        # x0 is top view, others are side views         
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)

        top1_up = self.topview_up0(y0)
        top2_up = self.topview_up1(y0)
        top3_up = self.topview_up2(y0)
        top4_up = self.topview_up3(y0)
        
        top1_down = self.topview_down0(y1)
        top2_down = self.topview_down1(y2)
        top3_down = self.topview_down2(y3)
        top4_down = self.topview_down3(y4)


        y21 = self.cross_conv21(y2)
        y41 = self.cross_conv41(y4)
        
        y12 = self.cross_conv12(y1)
        y32 = self.cross_conv32(y3)
        
        y23 = self.cross_conv23(y2)
        y43 = self.cross_conv43(y4)
        
        y14 = self.cross_conv14(y1)
        y34 = self.cross_conv34(y3)
    
        
        out0 = out0 + 0.25 * (top1_down + top2_down + top3_down + top4_down)
        out1 = out1 + (1 / 3) * (top1_up + y21 + y41)
        out2 = out2 + (1 / 3) * (top2_up + y12 + y32)
        out3 = out3 + (1 / 3) * (top3_up + y23 + y43)
        out4 = out4 + (1 / 3) * (top4_up + y14 + y34)
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
            
        to_repermute = [out0,out1,out2,out3,out4]   
        out0, out1, out2, out3, out4 = [to_repermute[self.perm_inv[i]] for i in range(5)]
        return out0, out1, out2, out3, out4
    
    
# same as naive implementation, but views 0 and 4 wrap around for symmetry
class WrappedCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(WrappedCrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = 'replicate'  # 'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.up_conv10 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv32 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv43 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        # this is the "wrapping" up conv
        self.up_conv04 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)

    
        self.down_conv01 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.down_conv12 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.down_conv23 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.down_conv34 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        # this is the "wrapping" down conv
        self.down_conv40 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)

        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2, x3, x4):
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)

        y0_up = self.up_conv04(y0)
        y1_up = self.up_conv10(y1)
        y2_up = self.up_conv21(y2)
        y3_up = self.up_conv32(y3)
        y4_up = self.up_conv43(y4)

        y0_down = self.down_conv01(y0)
        y1_down = self.down_conv12(y1)
        y2_down = self.down_conv23(y2)
        y3_down = self.down_conv34(y3)
        y4_down = self.down_conv40(y4)

        out0 = out0 + y4_down + y1_up 
        out1 = out1 + y0_down + y2_up
        out2 = out2 + y1_down + y3_up
        out3 = out3 + y2_down + y4_up
        out4 = out4 + y3_down + y0_up
        
        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4
    
    
    
# no cross-view connections; each view stays within it's own flow
class SeparatedConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(SeparatedConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = 'replicate'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))

        # conv2D forward
        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        
        
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale3_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale4_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2, x3, x4):
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)
        out3 = self.conv_scale3_0(x3)
        out4 = self.conv_scale4_0(x4)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)
        y3 = self.lr(out3)
        y4 = self.lr(out4)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)
        out3 = self.conv_scale3_1(y3)
        out4 = self.conv_scale4_1(y4)


        out0 = out0 
        out1 = out1 
        out2 = out2 
        out3 = out3
        out4 = out4 

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
            out3 = out3 * self.gamma3
            out4 = out4 * self.gamma4
        return out0, out1, out2, out3, out4


class ParallelPermute(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, seed):
        super(ParallelPermute, self).__init__()
        self.n_inputs = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]

        np.random.seed(seed)
        perm, perm_inv = self.get_random_perm(0)
        self.perm = [perm]
        self.perm_inv = [perm_inv]

        for i in range(1, self.n_inputs):
            perm, perm_inv = self.get_random_perm(i)
            self.perm.append(perm)
            self.perm_inv.append(perm_inv)

    def get_random_perm(self, i):
        perm = np.random.permutation(self.in_channels[i])
        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i

        perm = torch.LongTensor(perm)
        perm_inv = torch.LongTensor(perm_inv)
        return perm, perm_inv

    def forward(self, x, rev=False):
        if not rev:
            return [x[i][:, self.perm[i]] for i in range(self.n_inputs)]
        else:
            return [x[i][:, self.perm_inv[i]] for i in range(self.n_inputs)]

    def jacobian(self, x, rev=False):
        return [0.] * self.n_inputs

    def output_dims(self, input_dims):
        return input_dims


class parallel_glow_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class, F_args={},
                 clamp=5., use_noise=False):
        super(parallel_glow_coupling_layer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp

        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)
        
        self.use_noise = use_noise
        self.cond_dim = 1 if self.use_noise else 0

        self.s1 = F_class(self.split_len1 + self.cond_dim, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2 + self.cond_dim, self.split_len1 * 2, **F_args)

    def e(self, s):
        if self.clamp > 0:
            return torch.exp(self.log_e(s))
        else:
            return torch.exp(s)

    def log_e(self, s):
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(s / self.clamp)
        else:
            return s
    
    def forward(self, x, rev=False):
        
        if self.use_noise:            
            def c(a,b):
                return torch.cat((a,b), dim=1)
            cond = x[5] 
        
        x01, x02 = (x[0].narrow(1, 0, self.split_len1),
                    x[0].narrow(1, self.split_len1, self.split_len2))
        x11, x12 = (x[1].narrow(1, 0, self.split_len1),
                    x[1].narrow(1, self.split_len1, self.split_len2))
        x21, x22 = (x[2].narrow(1, 0, self.split_len1),
                    x[2].narrow(1, self.split_len1, self.split_len2))
        x31, x32 = (x[3].narrow(1, 0, self.split_len1),
                    x[3].narrow(1, self.split_len1, self.split_len2))
        x41, x42 = (x[4].narrow(1, 0, self.split_len1),
                    x[4].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            if self.use_noise:
                r02, r12, r22, r32, r42 = self.s2(c(x02, cond), c(x12, cond), c(x22, cond), c(x32, cond), c(x42, cond))
            else:
                r02, r12, r22, r32, r42 = self.s2(x02, x12, x22, x32, x42)

            s02, t02 = r02[:, :self.split_len1], r02[:, self.split_len1:]
            s12, t12 = r12[:, :self.split_len1], r12[:, self.split_len1:]
            s22, t22 = r22[:, :self.split_len1], r22[:, self.split_len1:]
            s32, t32 = r32[:, :self.split_len1], r32[:, self.split_len1:]
            s42, t42 = r42[:, :self.split_len1], r42[:, self.split_len1:]

            y01 = self.e(s02) * x01 + t02
            y11 = self.e(s12) * x11 + t12
            y21 = self.e(s22) * x21 + t22
            y31 = self.e(s32) * x31 + t32
            y41 = self.e(s42) * x41 + t42

            if self.use_noise:
                r01, r11, r21, r31, r41 = self.s1(c(y01, cond), c(y11, cond), c(y21, cond), c(y31, cond), c(y41, cond))
            else:
                r01, r11, r21, r31, r41 = self.s1(y01, y11, y21, y31, y41)

            s01, t01 = r01[:, :self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, :self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, :self.split_len2], r21[:, self.split_len2:]
            s31, t31 = r31[:, :self.split_len2], r31[:, self.split_len2:]
            s41, t41 = r41[:, :self.split_len2], r41[:, self.split_len2:]
            y02 = self.e(s01) * x02 + t01
            y12 = self.e(s11) * x12 + t11
            y22 = self.e(s21) * x22 + t21
            y32 = self.e(s31) * x32 + t31
            y42 = self.e(s41) * x42 + t41

        else:  # names of x and y are swapped!
            raise NotImplementedError("Reverse is not needed for inference in AD; therefore it's not implemented!")
            r01, r11, r21 = self.s1(x01, x11, x21)

            s01, t01 = r01[:, :self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, :self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, :self.split_len2], r21[:, self.split_len2:]

            y02 = (x02 - t01) / self.e(s01)
            y12 = (x12 - t11) / self.e(s11)
            y22 = (x22 - t21) / self.e(s21)

            r02, r12, r22 = self.s2(y02, y12, y22)

            s02, t02 = r02[:, :self.split_len2], r01[:, self.split_len2:]
            s12, t12 = r12[:, :self.split_len2], r11[:, self.split_len2:]
            s22, t22 = r22[:, :self.split_len2], r21[:, self.split_len2:]

            y01 = (x01 - t02) / self.e(s02)
            y11 = (x11 - t12) / self.e(s12)
            y21 = (x21 - t22) / self.e(s22)

        y0 = torch.cat((y01, y02), 1)
        y1 = torch.cat((y11, y12), 1)
        y2 = torch.cat((y21, y22), 1)
        y3 = torch.cat((y31, y32), 1)
        y4 = torch.cat((y41, y42), 1)

        y0 = torch.clamp(y0, -1e6, 1e6)
        y1 = torch.clamp(y1, -1e6, 1e6)
        y2 = torch.clamp(y2, -1e6, 1e6)
        y3 = torch.clamp(y3, -1e6, 1e6)
        y4 = torch.clamp(y4, -1e6, 1e6)
        
        # new version, fitting one gaussian per pixel
        jac0 = torch.sum(self.log_e(s01), dim=(1,)) + torch.sum(self.log_e(s02), dim=(1,))
        jac1 = torch.sum(self.log_e(s11), dim=(1,)) + torch.sum(self.log_e(s12), dim=(1,))
        jac2 = torch.sum(self.log_e(s21), dim=(1,)) + torch.sum(self.log_e(s22), dim=(1,))
        jac3 = torch.sum(self.log_e(s31), dim=(1,)) + torch.sum(self.log_e(s32), dim=(1,))
        jac4 = torch.sum(self.log_e(s41), dim=(1,)) + torch.sum(self.log_e(s42), dim=(1,))
        self.last_jac = [jac0, jac1, jac2, jac3, jac4]

        return [y0, y1, y2, y3, y4]

    def jacobian(self, x, rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims
