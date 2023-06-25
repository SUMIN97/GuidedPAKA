# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from .guidedpaka import GuidedPAKA2d
from .head import build_neighbor_head, build_semantic_head

__all__ = ["GuidedPAKAFPN"]


class GuidedPAKAFPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
            self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", cfg=None
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): ['res2', 'res3', 'res4', 'res5']
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(GuidedPAKAFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]  # [4,8,16,32]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]  # [256, 512, 1024, 2048]

        self.nei_head = build_neighbor_head(cfg)
        self.sem_head = build_semantic_head(cfg)

        self.nei_conv = Conv2d(len(cfg.MODEL.NEIGHBOR_HEAD.DILATIONS) * cfg.MODEL.NEIGHBOR_HEAD.NUM_NEIGHBOR, 9,
                               kernel_size=1)
        self.sem_conv = Conv2d(
            cfg.MODEL.SEMANTIC_HEAD.NUM_CLASSES,
            cfg.MODEL.SEMANTIC_FPN.CONVS_DIM,
            kernel_size=1)

        self.output_strides = cfg.MODEL.GUIDANCE_OUTPUT_STRIDES[::-1]  # [32,16,8]

        for layer in [self.sem_conv]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            if idx == 0:  # res2
                output_conv = Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                )
            else:  # res3, res4, res5
                output_conv = GuidedPAKA2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm
                )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]  # [res5, res4, res3, res2]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        pred_semantics, pred_neighbors = [], []

        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        feat, pd_sem, pd_nei = self.paka_control_range(0, prev_features)
        results.append(feat)
        pred_semantics.append(pd_sem)
        pred_neighbors.append(pd_nei)

        # Reverse feature maps into top-down order (from low to high resolution)
        # res5 : idx0, res4:idx1, res3:idx2

        for idx, (lateral_conv, output_conv) in enumerate(
                zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2

                ## 내가 추가 ##
                if idx == 3:  # p2
                    results.insert(0, output_conv(prev_features))
                else:  # p3,p4,p5
                    # feat, pd_sem, pd_nei = self.paka_output_conv(idx, prev_features)
                    # print('idx', prev_features.shape)
                    feat, pd_sem, pd_nei = self.paka_control_range(idx, prev_features)
                    results.insert(0, feat)
                    pred_semantics.append(pd_sem)
                    pred_neighbors.append(pd_nei)

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}, pred_semantics, pred_neighbors

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def paka_output_conv(self, idx, feat):
        pd_sem = self.sem_head(feat)
        # guide_ch = torch.sigmoid(pd_sem)
        # a = guide_ch[0, :, 10,10]
        # print('ch', a.max().item(), a.min().item())
        # guide_ch = torch.exp(pd_sem)
        # guide_ch = torch.softmax(guide_ch, dim=1)
        guide_ch = self.sem_conv(pd_sem)

        # guide_ch = self.sem_norm(guide_ch)

        pd_nei = self.nei_head(feat)
        # guide_sp = torch.sigmoid(pd_nei)
        # a = guide_sp[0, :, 10, 10]
        # print('sp', a.max().item(), a.min().item())
        # guide_sp = self.nei_conv(guide_sp)
        # guide_sp = self.nei_norm(guide_sp)

        feat = self.output_convs[idx](feat, guide_ch, pd_nei)

        return feat, pd_sem, pd_nei

    def paka_control_range(self, idx, feat):
        pd_sem = self.sem_head(feat)
        guide_ch = torch.sigmoid(pd_sem)  # [0,1]
        guide_ch = self.sem_conv(guide_ch)

        pd_nei = self.nei_head(feat)
        guide_sp = torch.sigmoid(pd_nei)  # [0,1]
        guide_sp = self.nei_conv(guide_sp)

        # print('idx', idx)
        feat = self.output_convs[idx](feat, guide_ch, guide_sp)

        return feat, pd_sem, pd_nei

    @torch.no_grad()
    def resize_input(self, idx, image):
        return F.interpolate(image, scale_factor=1 / self.output_strides[idx], mode='bilinear', align_corners=False)


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )









