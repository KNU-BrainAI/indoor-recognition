# This is the main training file we are using
import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

from datasets import Dataset4YoloAngle
from utils import MWtools, timer, visualization
import api
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rapid_pL1')
    parser.add_argument('--backbone', type=str, default='dark53')
    parser.add_argument('--dataset', type=str, default='HBCP')
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--high_resolution', action='store_true')

    parser.add_argument('--checkpoint', type=str, default='')

    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--img_interval', type=int, default=500)
    parser.add_argument('--print_interval', type=int, default=500)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    
    parser.add_argument('--debug', action='store_true') # default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert torch.cuda.is_available() # Currently do not support CPU training
    # -------------------------- settings ---------------------------
    target_size = 480 if args.high_resolution else 480
    initial_size = 480 if args.high_resolution else 480
    job_name = f'{args.model}_{args.backbone}_{args.dataset}{target_size}'
    # dataloader setting
    batch_size = args.batch_size
    num_cpu = 0 if batch_size == 1 else 16
    subdivision = 128 // batch_size
    enable_aug = True
    multiscale = True
    multiscale_interval = 10
    # SGD optimizer
    decay_SGD = 0.0005 * batch_size * subdivision
    print(f'effective batch size = {batch_size} * {subdivision}')
    # dataset setting
    print('initialing dataloader...')
    
    if args.dataset == 'HBCP':
        videos = ['Meeting1', 'All_off', 'IRfilter',
                  'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases', 'IRill', 'High_activity', 'COCO_train2017']
        # if args.high_resolution:
        #     videos += ['All_off', 'IRfilter', 'IRill']
        train_img_dir = [f'/home/user/4TB/hwlee/COSSY/{s}/' for s in videos]
        train_json = [f'/home/user/4TB/hwlee/COSSY/annotations/{s}.json' for s in videos]
        val_img_dir = '/home/user/4TB/hwlee/COSSY/Lab1/'
        val_json = '/home/user/4TB/hwlee/COSSY/annotations/Lab1.json'
        lr_SGD = 0.0001 / batch_size / subdivision
        # print('lr_SGD:', lr_SGD)
        # Learning rate setup
        def burnin_schedule(i):
            burn_in = 500
            if i < burn_in:
                factor = (i / burn_in) ** 2
            elif i < 10000:
                factor = 1.0
            elif i < 20000:
                factor = 0.3
            else:
                factor = 0.1
            return factor
            
    dataset = Dataset4YoloAngle(train_img_dir, train_json, initial_size, enable_aug,
                                only_person=True, debug_mode=args.debug)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
    dataiterator = iter(dataloader)
    
    if args.model == 'rapid_pL1':
        from models.rapid import RAPiD
        model = RAPiD(backbone=args.backbone, img_norm=False,
                       loss_angle='period_L1')
        
    elif args.model == 'rapid_pL2':
        from models.rapid import RAPiD
        model = RAPiD(backbone=args.backbone, img_norm=False,
                       loss_angle='period_L2')
    
    model = model.cuda()

    start_iter = -1
    if args.checkpoint:
        print("loading ckpt...", args.checkpoint)
        weights_path = os.path.join('./weights/', args.checkpoint)
        state = torch.load(weights_path)
        model.load_state_dict(state['model'])
        start_iter = state['iter']
        

    val_set = MWtools.MWeval(val_json, iou_method='rle')
    eval_img_names = os.listdir('./images/')
    eval_img_paths = [os.path.join('./images/',s) for s in eval_img_names if s.endswith('.jpg')]
    logger = SummaryWriter(f'./logs/{job_name}')

    # optimizer setup
    params = []
    # set weight decay only on conv.weight
    for key, value in model.named_parameters():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay_SGD}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]

    if args.dataset == 'COCO':
        optimizer = torch.optim.SGD(params, lr=lr_SGD, momentum=0.9, dampening=0,
                                    weight_decay=decay_SGD)
    elif args.dataset in {'MW', 'HBCP', 'HBMW', 'CPMW'}:
        assert args.checkpoint is not None
        optimizer = torch.optim.SGD(params, lr=lr_SGD)
    else:
        raise NotImplementedError()

    if args.dataset not in args.checkpoint:
        start_iter = -1
    else:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f'begin from iteration: {start_iter}')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule, last_epoch=start_iter)
    
    # start training loop
    today = timer.today()
    start_time = timer.tic()
    for iter_i in tqdm(range(start_iter, 20000)):
        # evaluation
        if iter_i % args.eval_interval == 0 and (iter_i > 0): # (args.dataset != 'COCO' or iter_i > 0)
            with timer.contexttimer() as t0:
                model.eval()
                model_eval = api.Detector(conf_thres=0.005, model=model)
                dts = model_eval.detect_imgSeq(val_img_dir, input_size=target_size)
                str_0 = val_set.evaluate_dtList(dts, metric='AP')
            s = f'\nCurrent time: [ {timer.now()} ], iteration: [ {iter_i} ]\n\n'
            s += str_0 + '\n\n'
            s += f'Validation elapsed time: [ {t0.time_str} ]'
            print(s)
            logger.add_text('Validation summary', s, iter_i)
            logger.add_scalar('Validation AP[IoU=0.5]', val_set._getAP(0.5), iter_i)
            logger.add_scalar('Validation AP[IoU=0.75]', val_set._getAP(0.75), iter_i)
            logger.add_scalar('Validation AP[IoU=0.5:0.95]', val_set._getAP(), iter_i)
            model.train()
            
        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            imgs = imgs.cuda()
            loss = model(imgs, targets, labels_cats=cats)
            loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % args.print_interval == 0:
            sec_used = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            avg_iter = timer.sec2str(sec_used/(iter_i+1-start_iter))
            avg_epoch = avg_iter / batch_size / subdivision * 118287
            print(f'\nTotal time: {time_used}, iter: {avg_iter}, epoch: {avg_epoch}')
            current_lr = scheduler.get_last_lr()[0] * batch_size * subdivision
            print(f'[Iteration {iter_i}] [learning rate {current_lr:.3g}]',
                  f'[Total loss {loss:.2f}] [img size {dataset.img_size}]')
            print(model.loss_str)
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print(f'Max GPU memory usage: {max_cuda} GigaBytes')
            torch.cuda.reset_peak_memory_stats(0)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', f'{job_name}_{today}_{iter_i}.ckpt')
            torch.save(state_dict, save_path)