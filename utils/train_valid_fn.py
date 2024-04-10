import os.path as osp

import torch
import torch.nn as nn


from models.losses import MSELoss

from mmpose.core.evaluation.top_down_eval import keypoint_pck_accuracy
from utils.transform import fliplr_joints, affine_transform, get_affine_transform

from torch.nn.utils import clip_grad_norm_

from torch.utils.data import DataLoader, Dataset

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from time import time
from timm.utils import accuracy, AverageMeter
from utils.utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, \
    auto_resume_helper, \
    reduce_tensor


from utils.logging import get_root_logger
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
import numpy as np


@torch.no_grad()
def valid_model(model: nn.Module, dataloaders: DataLoader, criterion: nn.Module, cfg: dict):
    logger = get_root_logger()
    total_loss = 0
    accuracy_all = 0
    total_metric = 0
    model.eval()
    for dataloader in dataloaders:
        train_pbar = tqdm(dataloader)
        for batch_idx, batch in enumerate(train_pbar):
            images, targets, target_weights, __ = batch
            images = images.to('cuda')
            targets = targets.to('cuda')
            target_weights = target_weights.to('cuda')

            if images.size(0) != 32:
                # 计算扩展的数量
                extension_size = 32 - images.size(0)
                # 在第一个维度上复制扩展
                images = torch.cat([images] * (extension_size + 1), dim=0)[:32]
            if targets.size(0) != 32:
                extension_size = 32 - targets.size(0)
                # 在第一个维度上复制扩展
                targets = torch.cat([targets] * (extension_size + 1), dim=0)[:32]

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            arr = np.full((32, 2), 224)
            mask = np.full((32, 17), True, dtype=bool)

            preds = outputs.detach().cpu().numpy()
            target = targets.detach().cpu().numpy()

            _, accuracy, _ = keypoint_pck_accuracy(preds, target, mask, 0.05, arr)
            accuracy_all += accuracy
            logger.info(f"accuracy {accuracy:.6f} "
                        f" Loss (val) {loss:.4f} ")
    avg_loss = total_loss / (len(dataloader) * len(dataloaders))
    avg_accuracy = accuracy_all / len(dataloader)

    return avg_loss, avg_accuracy


def train_model(model: nn.Module, datasets_train: Dataset, datasets_valid: Dataset, cfg: dict, distributed: bool,
                validate: bool, train: bool, timestamp: str,
                meta: dict) -> None:
    logger = get_root_logger()

    num_samples = len(datasets_train)
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
    # Prepare data loaders
    '''datasets_train = datasets_train if isinstance(datasets_train, (list, tuple)) else [datasets_train]
    datasets_valid = datasets_valid if isinstance(datasets_valid, (list, tuple)) else [datasets_valid]'''

    '''if distributed:
        samplers_train = [
            DistributedSampler(ds, num_replicas=len(cfg.gpu_ids), rank=torch.cuda.current_device(), shuffle=True,
                               drop_last=False) for ds in datasets_train]
        samplers_valid = [DistributedSampler(ds, num_replicas=len(cfg.gpu_ids), rank=torch.cuda.current_device(), shuffle=True, drop_last=False) for ds in datasets_valid]
    else:'''
    '''samplers_train = [None for ds in datasets_train]
    samplers_valid = [None for ds in datasets_valid]'''

    '''dataloaders_train = [DataLoader(ds, batch_size=cfg.data['samples_per_gpu'], shuffle=False, sampler=sampler,
                                    num_workers=cfg.data['workers_per_gpu'], pin_memory=False) for ds, sampler in
                         zip(datasets_train, samplers_train)]'''
    # dataloaders_valid = [DataLoader(ds, batch_size=cfg.data['samples_per_gpu'], shuffle=False, sampler=sampler, num_workers=cfg.data['workers_per_gpu'], pin_memory=False) for ds, sampler in zip(datasets_valid, samplers_valid)]
    dataloaders_train = DataLoader(datasets_train, batch_size=cfg.data['samples_per_gpu'], shuffle=False,
                                   num_workers=cfg.data['workers_per_gpu'], pin_memory=False)
    dataloaders_valid = DataLoader(datasets_valid, batch_size=cfg.data['samples_per_gpu'], shuffle=False,
                                   num_workers=cfg.data['workers_per_gpu'], pin_memory=False)
    # put model on gpus
    '''if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        model = DistributedDataParallel(
            module=model, 
            device_ids=[torch.cuda.current_device()], 
            broadcast_buffers=False, 
            find_unused_parameters=find_unused_parameters)
    else:
        model = DataParallel(model, device_ids=cfg.gpu_ids)'''

    model = model.to('cuda')
    model_without_ddp = model
    # Loss function
    criterion = MSELoss()
    # criterion = JointsMSELoss(use_target_weight=cfg.model['keypoint_head']['loss_keypoint']['use_target_weight'])
    max_accuracy = 0.0
    # Optimizer
    # optimizer = AdamW(model.parameters(), lr=cfg.optimizer['lr'], betas=cfg.optimizer['betas'], weight_decay=cfg.optimizer['weight_decay'])
    optimizer = build_optimizer(cfg, model)
    loss_scaler = NativeScalerWithGradNormCount()

    lr_scheduler = build_scheduler(cfg, optimizer, len(dataloaders_train))

    if cfg.TRAIN['AUTO_RESUME']:
        resume_file = auto_resume_helper(cfg.OUTPUT)
        if resume_file:
            if cfg.MODEL['RESUME']:
                logger.warning(f"auto-resume changing resume file from {cfg.MODEL['RESUME']} to {resume_file}")
            cfg.MODEL['RESUME'] = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.OUTPUT}, ignoring auto resume')

    if cfg.MODEL['RESUME']:
        max_accuracy = load_checkpoint(cfg, model, optimizer, lr_scheduler, loss_scaler, logger)
        # avg_loss_valid,avg_accuracy = valid_model(model, dataloaders_valid, criterion, cfg)
        # logger.info(f"Accuracy of the network on the {len(dataloaders_valid[0])} test images: {avg_accuracy:.5f}")

    # Layer-wise learning rate decay
    '''lr_mult = [cfg.optimizer['paramwise_cfg']['layer_decay_rate']] * cfg.optimizer['paramwise_cfg']['num_layers']
    layerwise_optimizer = LayerDecayOptimizer(optimizer, lr_mult)'''

    # Learning rate scheduler (MultiStepLR)
    '''milestones = cfg.lr_config['step']
    gamma = 0.1
    scheduler = MultiStepLR(optimizer, milestones, gamma)
'''
    # Warm-up scheduler
    '''num_warmup_steps = cfg.lr_config['warmup_iters']  # Number of warm-up steps
    warmup_factor = cfg.lr_config['warmup_ratio']  # Initial learning rate = warmup_factor * learning_rate
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_factor + (1.0 - warmup_factor) * step / num_warmup_steps
    )'''

    # AMP setting
    if cfg.use_amp:
        logger.info("Using Automatic Mixed Precision (AMP) training...")
        # Create a GradScaler object for FP16 training
        scaler = GradScaler()

    # Logging config
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'''\n
    #========= [Train Configs] =========#
    # - Num GPUs: {len(cfg.gpu_ids)}
    # - Batch size (per gpu): {cfg.data['samples_per_gpu']}
    # - BASE_LR: {cfg.TRAIN['BASE_LR']: .10f}
    # - WARMUP_EPOCHS: {cfg.TRAIN['WARMUP_EPOCHS']: .6f}
    # - DECAY_EPOCHS: {cfg.TRAIN['LR_SCHEDULER']['DECAY_EPOCHS']: .6f}
    # - MIN_LR: {cfg.TRAIN['MIN_LR']: .15f}
    # - WARMUP_LR: {cfg.TRAIN['WARMUP_LR']: .15f}
    # - Num params: {total_params:,d}
    # - AMP: {cfg.use_amp}
    #===================================# 
    ''')

    global_step = 0

    # lr_scheduler = build_scheduler(cfg, optimizer, len(dataloader))
    for epoch in range(cfg.TRAIN['START_EPOCH'], cfg.TRAIN['EPOCHS']):
        model.train()
        train_pbar = tqdm(dataloaders_train)
        total_loss = 0
        accuracy_all = 0
        tic = time()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()
        if train:
            for batch_idx, batch in enumerate(train_pbar):
                optimizer.zero_grad()

                images, targets, target_weights, meta = batch
                num_images = images.size(0)
                '''if images.size(0) != 128:
                        # 计算扩展的数量
                        extension_size = 128 - images.size(0)
                        # 在第一个维度上复制扩展
                        images = torch.cat([images] * (extension_size + 1), dim=0)[:128]
                    if targets.size(0) != 128:
                        extension_size = 128 - targets.size(0)
                        # 在第一个维度上复制扩展
                        targets = torch.cat([targets] * (extension_size + 1), dim=0)[:128]'''

                images = images.to('cuda')
                targets = targets.to('cuda')

                target_weights = target_weights.to('cuda')

                if cfg.use_amp:
                    with autocast():
                        outputs = model(images)
                        '''print(outputs.shape)
                        print(targets.shape)'''

                        loss = criterion(outputs, targets, target_weights)

                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    grad_norm = loss_scaler(loss, optimizer, clip_grad=5,
                                            parameters=model.parameters(), create_graph=is_second_order,
                                            update_grad=True)

                    loss_scale_value = loss_scaler.state_dict()["scale"]
                    norm_meter.update(grad_norm)
                    scaler_meter.update(loss_scale_value)

                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    clip_grad = 5
                    clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()

                lr_scheduler.step_update(epoch * len(dataloaders_train) + batch_idx)

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
                r = meta['rotation'].numpy()

                preds = outputs.detach().cpu().numpy()
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
                idx += num_images
                # name_values, perf_indicator = datasets_valid.evaluate( all_preds, output_dir, all_boxes, image_path,filenames, imgnums)
                name.extend(meta['file_name'])

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
                '''if global_step < num_warmup_steps:
                        warmup_scheduler.step()'''
                global_step += 1
                # arr = np.full((128, 2), 224)
                # mask = np.full((128, 17), True, dtype=bool)

                # _, accuracy, _ = keypoint_pck_accuracy(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), mask, 0.01, arr)  # accuracy {accuracy:.4f}

                total_loss += loss.item()
                accuracy_all += accuracy
                logger.info(f"Epoch[{str(epoch).zfill(3)}/{str(cfg.TRAIN['EPOCHS']).zfill(3)}] "
                            f"Loss {loss.item():.4f}  "
                            f"LR {optimizer.param_groups[0]['lr']:.15f} "
                            f"accuracy {accuracy:.6f} "
                            f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t '
                            f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                            )
            '''scheduler.step()'''
            # lr_scheduler.step()

            avg_loss_train = total_loss / len(dataloaders_train)
            avg_accuracy = accuracy_all / len(dataloaders_train)
            logger.info(f"[Summary-train] Epoch [{str(epoch).zfill(3)}/{str(cfg.TRAIN['EPOCHS']).zfill(3)}] "
                        f"avg_accuracy {avg_accuracy:.6f} "
                        f"Average Loss (train) {avg_loss_train:.4f} "
                        f"{time() - tic:.5f} sec. elapsed ")

        name_values, perf_indicator = datasets_train.evaluate(all_preds, output_dir, all_boxes, image_path, filenames,imgnums)

        '''ckpt_name = f"epoch{str(epoch).zfill(3)}.pth"
            ckpt_path = osp.join(cfg.work_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)'''
        # validation
        if validate:
            tic2 = time()
            avg_loss_valid, avg_accuracy = valid_model(model, dataloaders_valid, criterion, cfg)
            max_accuracy = max(max_accuracy, avg_accuracy)

            logger.info(f"[Summary-valid] Epoch [{str(epoch).zfill(3)}/{str(cfg.TRAIN['EPOCHS']).zfill(3)}] "
                        f"Average Loss (valid) {avg_loss_valid:.4f} "
                        f"avg_accuracy {avg_accuracy:.6f} "
                        f"{time() - tic2:.5f} sec. elapsed ")

        '''save_checkpoint(cfg, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)'''

        save_state = {'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_scheduler.state_dict(),
                      'max_accuracy': max_accuracy,
                      'scaler': loss_scaler.state_dict(),
                      'epoch': epoch,
                      }

        save_path = osp.join(cfg.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        # if epoch % 10 == 0:
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")
