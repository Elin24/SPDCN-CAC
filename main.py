import sys

print(sys.executable)

import os
import time
import argparse
import datetime
import numpy as np
import subprocess
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from datasets import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, plot_curve, set_seed

def get_args_parser():
    parser = argparse.ArgumentParser('Counting Everything training and evaluation script', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--device', default='cuda:0', help='device name')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main_worker(config):
    data_loader_train, data_loader_val = build_loader(config.DATA, mode='train'), build_loader(config.DATA, mode='val')

    logger.info(f"Creating model: {config.MODEL.NAME}")
    model, criterion = build_model(config.MODEL)
    model.cuda()
    criterion.cuda()

    optimizer = build_optimizer(config, model)
    model_without_ddp = model


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    max_accuracy = [1e6] * 3

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        mae, mse, loss = validate(config, data_loader_val, model, criterion)
        max_accuracy = (mae, mse, loss)
        logger.info(f"Accuracy of the network on the test images: {mae:.2f} | {mse:.2f}")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    maestack, msestack, lossstack = [], [], []
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler)

        mae, mse, loss = validate(config, data_loader_val, model, criterion)
        maestack.append(mae)
        msestack.append(mse)
        lossstack.append(loss)
        plot_curve('mae', maestack, os.path.join('exp', config.TAG, 'train.log', 'mae_curve.png'))
        plot_curve('mse', msestack, os.path.join('exp', config.TAG, 'train.log', 'mse_curve.png'))
        plot_curve('loss', lossstack, os.path.join('exp', config.TAG, 'train.log', 'loss_curve.png'))

        logger.info(f"Accuracy of the network on the test images: {loss:.6f}")

        if mae < max_accuracy[0]:
            if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
                save_checkpoint(config, "best", model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
            max_accuracy = (mae, mse, loss)
        logger.info(f'Min total MAE|MSE|Loss: {max_accuracy[0]:.6f} | {max_accuracy[1]:.2f} | {max_accuracy[2] * 1e5:.2f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, boxes, targets, imgids) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        boxes = boxes.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        sdenmap = model(samples, boxes)
        bsize = torch.stack((boxes[:, 4] - boxes[:, 2], boxes[:, 3] - boxes[:, 1]), dim=-1)
        bs_mean = bsize.view(-1, 3, 2).float().mean(dim=1)
        loss = criterion(sdenmap, targets, box_size=bs_mean)
        
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr*1e5:.3f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val * 1e3 :.3f} ({loss_meter.avg *1e3 :.3f})\t'
                f'grad_norm {norm_meter.val * 1e2 :.3f} ({norm_meter.avg * 1e2 :.3f})\t'
                f'mem {memory_used:.0f}MB')
            
            # if smask is not None:
            #     logger.info(f'den_loss={loss_den.item()} | mask_loss={loss_mask.item()}')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    mae_meter = AverageMeter()
    mse_meter = AverageMeter()


    end = time.time()
    for idx, (images, boxes, target, imgids) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        boxes = boxes.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        bsize = target.size(0)
        # compute output
        with torch.no_grad():
            output = model(images, boxes)
            output = F.relu(output, inplace=True)
            output = output / config.MODEL.FACTOR
            tarsum = target.sum(dim=(1,2,3))
        
        loss = criterion(output * config.MODEL.FACTOR, target)
        diff = torch.abs(output.sum(dim=(1, 2, 3)) - tarsum)
        mae, mse = diff.mean(), (diff ** 2).mean()

        loss_meter.update(loss.item(), bsize)
        mae_meter.update(mae.item(), bsize)
        mse_meter.update(mse.item(), bsize)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'Loss {loss_meter.val:.6f} ({loss_meter.avg:.6f})  '
                f'MAE {mae_meter.val:.3f} ({mae_meter.avg:.3f})  '
                f'MSE {mse_meter.val ** 0.5:.3f} ({mse_meter.avg ** 0.5:.3f})  '
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * MAE {mae_meter.avg:.3f} MSE {mse_meter.avg ** 0.5:.3f}')
    return mae_meter.avg, mse_meter.avg ** 0.5, loss_meter.avg

if __name__ == '__main__':
    #torch.cuda.set_per_process_memory_fraction(0.5, 0)
    args, config = get_args_parser()

    
    torch.cuda.set_device(args.device)
    set_seed(config.SEED)

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        config.defrost()
        config.TRAIN.BASE_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.WARMUP_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.TRAIN.MIN_LR *= config.TRAIN.ACCUMULATION_STEPS
        config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main_worker(config)
