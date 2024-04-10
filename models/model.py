import torch.nn as nn

from .backbone.vit import ViT
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from .backbone.swin_transformer import SwinTransformer

from .backbone.swin_space import SwinTransformerspace
from .backbone.swin_time import PoseTransformer
import torch

__all__ = ['SwinPose']


class SwinPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(SwinPose, self).__init__()

        backbone_cfg = {k: v for k, v in cfg.model['backbone'].items() if k != 'type'}

        self.backbone = SwinTransformerspace(**backbone_cfg)
        self.keypoint_head = PoseTransformer()

    def forward(self, x):
        return self.keypoint_head(self.backbone(x))