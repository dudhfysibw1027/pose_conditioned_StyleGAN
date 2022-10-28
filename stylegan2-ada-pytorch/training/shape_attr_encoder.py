import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer, build_upsample_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.utils import get_root_logger


#----------------------------------------------------------------------------
class ShapeAttrEmbedding(nn.Module):

    def __init__(self, dim, out_dim, cls_num_list):
        super(ShapeAttrEmbedding, self).__init__()

        for idx, cls_num in enumerate(cls_num_list):
            setattr(
                self, f'attr_{idx}',
                nn.Sequential(
                    nn.Linear(cls_num, dim), nn.LeakyReLU(),
                    nn.Linear(dim, dim)))
        self.cls_num_list = cls_num_list
        self.attr_num = len(cls_num_list)
        self.fusion = nn.Sequential(
            nn.Linear(dim * self.attr_num, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, attr):
        attr_embedding_list = []
        for idx in range(self.attr_num):
            attr_embed_fc = getattr(self, f'attr_{idx}')
            attr_embedding_list.append(
                attr_embed_fc(
                    F.one_hot(
                        attr[:, idx],
                        num_classes=self.cls_num_list[idx]).to(torch.float32)))
        attr_embedding = torch.cat(attr_embedding_list, dim=1)
        attr_embedding = self.fusion(attr_embedding)

        return attr_embedding


#----------------------------------------------------------------------------
class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.
    This module consists of several plain convolutional layers.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convoluton in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out