import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule
from ..builder import NECKS


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dcn_v2 import DCN as dcn_v2




class SpatialAttention(nn.Module):
    def __init__(self,
                 conv_cfg=None,
                 #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 norm_cfg=dict(type='BN'),
                 act_cfg=None,
                 kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = ConvModule(2, 1, kernel_size, padding=padding, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SEM(nn.Module):
    def __init__(self,
                 in_chan,
                 out_chan,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 #norm_cfg=dict(type='BN'),
                 act_cfg=None):
        super(SEM, self).__init__()
        # self.conv_atten = ConvModule(in_chan, in_chan, kernel_size=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.sigmoid = nn.Sigmoid()
        self.sa = SpatialAttention()
        #self.ca = CoordAtt(in_chan, in_chan)
        self.conv = ConvModule(in_chan, out_chan, kernel_size=1, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # weight_init.c2_xavier_fill(self.conv_atten)
        # weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):

        feat_sa = self.sa(x)
        feat_sa = torch.mul(x, feat_sa)
        x = x + feat_sa

        feat = self.conv(x)
        return feat




class FeatureAlign_R2_new3(nn.Module):
    def __init__(self,
                 in_nc,
                 out_nc,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(FeatureAlign_R2_new3, self).__init__()

        self.lateral_conv = SEM(in_nc, out_nc)

        self.R2_lateral_conv = SEM(in_nc*2, out_nc)


        self.offset5 = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dcpack_L5 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.offset4 = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dcpack_L4 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        # self.offset3 = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.dcpack_L3 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                         extra_offset_mask=True)
        self.conv1x1_1 = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        #self.conv1x1_2 = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, *inputs, main_path=None):

        feat_l, feat_3, feat_4, feat_5 = inputs


        HW4 = feat_4.size()[2:]

        feat_up5 = F.interpolate(feat_5, HW4, mode='bilinear', align_corners=False)
        offset5 = self.offset5(torch.cat([feat_4, feat_up5 * 2], dim=1))
        feat_5_align = self.relu(self.dcpack_L5([feat_up5, offset5], main_path))
        #feat_4 = feat_5_align + feat_4
        feat_4 = torch.cat([feat_5_align, feat_4], dim=1)
        feat_4 = self.conv1x1_1(feat_4)

        HW3 = feat_3.size()[2:]

        feat_up4 = F.interpolate(feat_4, HW3, mode='bilinear', align_corners=False)
        offset4 = self.offset4(torch.cat([feat_3, feat_up4 * 2], dim=1))
        feat_4_align = self.relu(self.dcpack_L4([feat_up4, offset4], main_path))
        #feat_3 = feat_4_align + feat_3
        feat_3 = torch.cat([feat_4_align, feat_3], dim=1)

        HW2 = feat_l.size()[2:]
        #
        feat_2 = F.interpolate(feat_3, HW2, mode='bilinear', align_corners=False)
        # offset3 = self.offset3(torch.cat([feat_l, feat_up3 * 2], dim=1))
        # feat_3_align = self.relu(self.dcpack_L3([feat_up3, offset3], main_path))
        # feat_2 = feat_3_align + feat_l

        feat_up = self.R2_lateral_conv(feat_2)

        return feat_up


class FeatureAlign_2(nn.Module):
    def __init__(self,
                 in_nc,
                 out_nc,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(FeatureAlign_2, self).__init__()
        self.lateral_conv = SEM(in_nc, out_nc)
        self.cat = ConvModule(out_nc * 3, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.offset = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True
                                      )
        self.relu = nn.ReLU(inplace=True)
        #weight_init.c2_xavier_fill(self.offset)

    def forward(self, *inputs, main_path=None):

        feat_l, feat_3, feat_4, feat_5 = inputs
        HW = feat_l.size()[2:]

        feat_up3 = F.interpolate(feat_3, HW, mode='bilinear', align_corners=False)
        feat_up4 = F.interpolate(feat_4, HW, mode='bilinear', align_corners=False)
        feat_up5 = F.interpolate(feat_5, HW, mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        feat_up = self.cat(torch.cat([feat_up3, feat_up4, feat_up5], dim=1))
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return feat_align + feat_arm

class FeatureAlign_3(nn.Module):
    def __init__(self,
                 in_nc,
                 out_nc,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(FeatureAlign_3, self).__init__()
        self.lateral_conv = SEM(in_nc, out_nc)
        #self.cat = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.conv1x1_1 = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.offset = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True
                                      )
        self.relu = nn.ReLU(inplace=True)
        #weight_init.c2_xavier_fill(self.offset)

    def forward(self, *inputs, main_path=None):

        feat_l, feat_4, feat_5 = inputs
        HW = feat_l.size()[2:]

        feat_up4 = F.interpolate(feat_4, HW, mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up4 * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up4, offset], main_path))  # [feat, offset]
        return feat_align + feat_arm



class FeatureAlign_4(nn.Module):
    def __init__(self,
                 in_nc,
                 out_nc,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(FeatureAlign_4, self).__init__()
        self.lateral_conv = SEM(in_nc, out_nc)


        self.offset = ConvModule(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, *inputs, main_path=None):

        feat_l, feat_s = inputs
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return feat_align + feat_arm





@NECKS.register_module()
class FPN_ALL(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(FPN_ALL, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        in_channels_0, in_channels_1, in_channels_2, in_channels_3 = in_channels

        self.P5_1 = SEM(in_channels_3, out_channels)

        self.P4_1 = FeatureAlign_4(in_channels_2, out_channels)

        self.P3_1 = FeatureAlign_3(in_channels_1, out_channels)

        self.P2_1 = FeatureAlign_R2_new3(in_channels_0, out_channels)

        self.fpn2_conv = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fpn3_conv = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fpn4_conv = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fpn5_conv = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.fpn_downconv = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)



    @auto_fp16()
    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5 = self.P5_1(C5)  ## 1x1 -- 256

        P5_x = self.fpn5_conv(P5)  ## 3x3 smooth


        P4 = self.P4_1(C4, P5)  ## 1x1 -- 256

        P4_x = self.fpn4_conv(P4)  ## 3x3 smooth


        P3 = self.P3_1(C3, P4, P5)  ## 1x1 --256

        P3_x = self.fpn3_conv(P3)  ## 3x3 smooth

        P2 = self.P2_1(C2, P3, P4, P5)  #rebulid

        P2_x = self.fpn2_conv(P2)

        P6_x = self.fpn_downconv(P5)  ## P5 down sample


        outs = P2_x, P3_x, P4_x, P5_x, P6_x

        return tuple(outs)
