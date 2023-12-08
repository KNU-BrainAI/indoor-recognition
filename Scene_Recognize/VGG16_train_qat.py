import os
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from SASceneNet_qat import SASceneNet
from SASceneNet_qat import QuantizableResnet18
from Libs.Datasets.MITIndoor67Dataset_gray import MITIndoor67Dataset
from Libs.Utils import utils
from Libs.Utils.torchsummary import torchsummary
import numpy as np
import yaml
from tqdm import tqdm
import random
from mit_semseg.models import ModelBuilder, SegmentationModule
import cv2
import copy
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


global USE_CUDA, classes, CONFIG
USE_CUDA = torch.cuda.is_available()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42) 


def validation_cuda(dataloader, model_1, set):
    model_1.to(torch.device('cuda')).eval()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    Predictions = np.zeros(len(dataloader))
    SceneGTLabels = np.zeros(len(dataloader))

    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['VAL']

    data_time_start = time.time()


    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                RGB_images_np = RGB_image.cpu().numpy().transpose(0, 2, 3, 1)
                gray_images_np = rgb_to_gray(RGB_images_np)
                sceneLabelGT = mini_batch['Scene Index'].cuda()

            RGB_image.cuda()
            model_1.cuda()     
            outputSceneLabel = model_1.forward(RGB_image)

            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Measure Top1, Top2 and Top5 accuracy
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

            # Update values
            # losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top2.update(prec2.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # Measure batch elapsed time
            batch_time.update(time.time() - start_time)

            if i == 134:
                print('Testing {} set batch: [{}/{}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg: {top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} (avg: {top2.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg: {top5.avg:.3f})'.
                      format(set, i, len(dataloader), set, batch_time=batch_time, loss=losses,
                             top1=top1, top2=top2, top5=top5))

        ClassTPDic = {'Top1': ClassTPs_Top1.cpu().numpy(),
                      'Top2': ClassTPs_Top2.cpu().numpy(), 'Top5': ClassTPs_Top5.cpu().numpy()}

        print('Elapsed time for {} set evaluation {time:.3f} seconds'.format(set, time=time.time() - data_time_start))
        print("")

        return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic


def rgb_to_gray(images):
    gray_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray_image)
    return np.stack(gray_images)

def process_segmentation_output_cuda(segmentation_output, gray_images, threshold=0.5, batch_size=10):
    thresholded_output = torch.where(segmentation_output > threshold, segmentation_output, torch.tensor(0.0).cuda())
    gray_images_with_channel = gray_images.expand(batch_size, 150, 224, 224)
    final_output = gray_images_with_channel * thresholded_output
    return final_output

def process_segmentation_output_cpu(segmentation_output, gray_images, threshold=0.5, batch_size=10):
    thresholded_output = torch.where(segmentation_output > threshold, segmentation_output, torch.tensor(0.0).cpu())
    gray_images_with_channel = gray_images.expand(batch_size, 150, 224, 224)
    final_output = gray_images_with_channel * thresholded_output
    return final_output


def train(dataloader, model_1, optimizer, scheduler=None, device='cuda'):

    best_val_acc = 0
    best_val_loss = 100

    best_model = None
    
    model_1.train()
    model_1.to(device)
    patient = 0
    for epoch in range(1, CONFIG['TRAINING']['EPOCHS']+1):
        model_1.train()
        train_loss = []
        print(f'train epoch : {epoch}, running')
        for i, (mini_batch) in tqdm(enumerate(dataloader)):
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                sceneLabelGT = mini_batch['Scene Index'].cuda()

                RGB_images_np = RGB_image.cpu().numpy().transpose(0, 2, 3, 1)
                gray_images_np = rgb_to_gray(RGB_images_np)

            optimizer.zero_grad()
            outputSceneLabel = model_1.forward(RGB_image)

            loss = model_1.loss(outputSceneLabel, sceneLabelGT)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_top1, _, _, _val_loss, _ = validation_cuda(val_loader, model_1, set='Validation')
        _train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val top1 ACC : [{_val_top1:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_top1)

        if best_val_acc < _val_top1:
            best_val_acc = _val_top1
            best_val_loss = _val_loss
            best_model = {
                'state_dict': model_1.state_dict(),
                'arch': model_1.__class__.__name__,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(best_model, f'best_model_test_epoch_{epoch}.pth')
            patient = 0
        else:
            patient+=1
        if patient == 20:
           break

    return best_model


os.environ["CUDA_VISIBLE_DEVICE"]='0'
cudnn.benchmark = torch.cuda.is_available()


CONFIG = yaml.safe_load(open("Config/config_VGG16_case2no4_qat.yaml", 'r'))


print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

traindir = 'Data/Datasets/case2_no4'
valdir = 'Data/Datasets/case2_no4'

train_dataset = MITIndoor67Dataset(traindir, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
                                            shuffle=False, num_workers=0, pin_memory=True)

val_dataset = MITIndoor67Dataset(valdir, "val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'], SemRGB=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], shuffle=False,
                                            num_workers=0, pin_memory=True)
test_dataset = MITIndoor67Dataset(valdir, "val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'], SemRGB=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], shuffle=False,
                                            num_workers=0, pin_memory=True)
classes = train_dataset.classes

print('Dataset loaded!')
print('Dataset Information:')
print('Train set. Size {}. Batch size {}. Nbatches {}'
      .format(len(train_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], len(train_loader)))
print('Validation set. Size {}. Batch size {}. Nbatches {}'
      .format(len(val_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], len(val_loader)))
print('Train set number of scenes: {}' .format(len(classes)))
print('Validation set number of scenes: {}' .format(len(classes)))

print('-' * 65)

if not os.path.exists(CONFIG['EXP']['OUTPUT_DIR']):
    os.makedirs(CONFIG['EXP']['OUTPUT_DIR'])


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from torchvision import models
from torchvision.models import VGG16_Weights


model_vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model_vgg.classifier[6] = nn.Linear(in_features=4096, out_features=6, bias=True)


def loss(self, x, target):
    self.criterion = nn.CrossEntropyLoss()

    assert (x.shape[0] == target.shape[0])
    loss = self.criterion(x, target.long())
    return loss

model_vgg.loss = loss.__get__(model_vgg)

model_vgg.cpu().train()
model_vgg.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.backends.quantized.engine = 'fbgemm'
torch.quantization.quantize_dynamic()

model_vgg.eval()
model_vgg.fuse_model()
model_parameters = filter(lambda p: p.requires_grad, model_vgg.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])


optimizer = torch.optim.Adagrad(model_vgg.parameters(), lr=CONFIG['TRAINING']['LR'],weight_decay=CONFIG['TRAINING']['WEIGHT_DECAY'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

model_vgg.to('cuda').train()
model = train(train_loader, model_vgg, optimizer, scheduler, device=device)

val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = validation_cuda(test_loader, model_vgg, set='Validation')

print('-' * 65)
print('Evaluation statistics:')

print('Validation results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5))
print("complete")
