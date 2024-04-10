import os.path as osp

import torch
import torch.nn as nn

from models.losses import JointsMSELoss
from models.optimizer import LayerDecayOptimizer

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from time import time
from timm.utils import accuracy, AverageMeter
from utils.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, \
    auto_resume_helper, \
    reduce_tensor
from models.losses import MSELoss
from mmpose.core.evaluation import (keypoint_pck_accuracy,
                                    keypoints_from_regression)

from utils.dist_util import get_dist_info, init_dist
from utils.logging import get_root_logger
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
import numpy as np
from utils.top_down_eval import keypoints_from_heatmaps
from utils.transform import fliplr_joints, affine_transform, get_affine_transform


@torch.no_grad()
def valid_model(model: nn.Module, datasets_valid: Dataset, dataloaders: DataLoader, criterion: nn.Module, cfg: dict):
    tic2 = time()
    logger = get_root_logger()
    total_loss = 0
    accuracy_all = 0
    total_metric = 0
    model.eval()
    num_samples = len(datasets_valid)
    print(num_samples)
    all_preds = np.zeros(
        (num_samples, 17, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    idx = 0
    image_path = []
    filenames = []
    imgnums = []
    name = []
    output_dir = r'D:\Pythoncoda\human3.6dataset\output'

    train_pbar = tqdm(dataloaders)

    for batch_idx, batch in enumerate(train_pbar):

        images, targets, target_weights, meta = batch
        num_images = images.size(0)
        # print(target_weights.shape)
        images = images.to('cuda')
        targets = targets.to('cuda')
        target_weights = target_weights.to('cuda')

        '''if images.size(0) != 32:
                # 计算扩展的数量
                extension_size = 32 - images.size(0)
                # 在第一个维度上复制扩展
                images = torch.cat([images] * (extension_size + 1), dim=0)[:32]
            if targets.size(0) != 32:
                extension_size = 32 - targets.size(0)
                # 在第一个维度上复制扩展
                targets = torch.cat([targets] * (extension_size + 1), dim=0)[:32]'''
        '''if images.size(0) != 32:
            # 计算扩展的数量
            extension_size = 32 - images.size(0)
            # 在第一个维度上复制扩展
            images = torch.cat([images] * (extension_size + 1), dim=0)[:32]
        if targets.size(0) != 32:
            extension_size = 32 - targets.size(0)
            # 在第一个维度上复制扩展
            targets = torch.cat([targets] * (extension_size + 1), dim=0)[:32]

        if target_weights.size(0) != 32:
            extension_size = 32 - target_weights.size(0)
            # 在第一个维度上复制扩展
            target_weights = torch.cat([target_weights] * (extension_size + 1), dim=0)[:32]'''

        outputs = model(images)

        loss = criterion(outputs, targets, target_weights)

        total_loss += loss.item()

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()
        r = meta['rotation'].numpy()

        preds = outputs.clone().cpu().numpy()
        for i in range(outputs.shape[0]):
            in_trans = get_affine_transform(c[i], s[i], 200, r[i], (224, 224), inv=1)
            for j in range(outputs.shape[1]):
                preds[i][j, 0:2] = affine_transform(preds[i][j, 0:2], in_trans)

        # points, prob = keypoints_from_heatmaps(outputs.clone().cpu().numpy(), c,s,unbiased=True, use_udp=True)
        # print(maxvals)
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = target_weights.clone().cpu().numpy()
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['imgId'])
        name.extend(meta['file_name'])
        idx += num_images
        # name_values, perf_indicator = datasets_valid.evaluate( all_preds, output_dir, all_boxes, image_path,filenames, imgnums)
        threshold_bbox = []

        for i in range(outputs.shape[0]):
            bbox = np.array((meta['bbox'][2][i], meta['bbox'][3][i]))
            bbox_thr = np.max(bbox)
            threshold_bbox.append(np.array([bbox_thr, bbox_thr]))

        threshold_bbox = np.array(threshold_bbox)
        _, accuracy, _ = keypoint_pck_accuracy(
            outputs.detach().cpu().numpy(),
            targets.detach().cpu().numpy(),
            target_weights[:, :, 0].detach().cpu().numpy() > 0,
            thr=0.05,
            normalize=threshold_bbox)
        # accuracy = model.keypoint_head.get_accuracy(outputs, targets.clone(), target_weights)
        # _, avg_acc, cnt, pred = accuracy1(outputs.detach().cpu().numpy(),targets.detach().cpu().numpy())

        accuracy_all += accuracy
        logger.info(f"Average Loss (valid) {loss:.4f} "
                    f"avg_accuracy {accuracy:.6f} "
                    f"{time() - tic2:.5f} sec. elapsed ")

    name_values, perf_indicator = datasets_valid.evaluate(all_preds, output_dir, all_boxes, image_path, filenames,imgnums)
    avg_loss_valid = total_loss / (len(dataloaders))
    avg_accuracy = accuracy_all / len(dataloaders)

    logger.info(f"Average Loss (valid) {avg_loss_valid:.4f} "
                f"avg_accuracy {avg_accuracy:.6f} "
                f"{time() - tic2:.5f} sec. elapsed ")

    return perf_indicator
