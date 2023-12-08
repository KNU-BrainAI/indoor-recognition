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
    model_1.cuda()
    model2.cuda()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()

    model_1.eval()
    model2.eval()

    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['VAL']

    data_time_start = time.time()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                RGB_images_np = RGB_image.cpu().numpy().transpose(0, 2, 3, 1)
                gray_images_np = rgb_to_gray(RGB_images_np)
                gray_images_tensor = torch.tensor(gray_images_np).unsqueeze(1).cuda()
                semantic_scores = model2(RGB_image, feed_dict=None, segSize=(224,224))
                sceneLabelGT = mini_batch['Scene Index'].cuda()
                gray_segmentation_tensor=process_segmentation_output_cuda(semantic_scores, gray_images_tensor, batch_size=len(gray_images_np), threshold=0.5)

            # Model Forward
            imgNscore = [RGB_image, gray_segmentation_tensor] 
            RGB_image.cuda()
            gray_segmentation_tensor.cuda()
            model_1.cuda()     
            outputSceneLabel, _, _, _ = model_1.forward_sascene(imgNscore)
            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = model_1.loss(outputSceneLabel, sceneLabelGT)

            # Measure Top1, Top2 and Top5 accuracy
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

            # Update values
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top2.update(prec2.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # Measure batch elapsed time
            batch_time.update(time.time() - start_time)

            # Print information
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

def validation_cpu(dataloader, model_1, set):

    model_1.cpu()
    model2.cpu()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cpu()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cpu()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cpu()

    model_1.eval()
    model2.eval()

    # Extract batch size
    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['VAL']

    # Start data time
    data_time_start = time.time()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            RGB_image = mini_batch['Image'].cpu()
            RGB_images_np = RGB_image.cpu().numpy().transpose(0, 2, 3, 1)
            gray_images_np = rgb_to_gray(RGB_images_np)
            gray_images_tensor = torch.tensor(gray_images_np).unsqueeze(1).cpu()
            semantic_scores = model2(RGB_image, feed_dict=None, segSize=(224,224))
            sceneLabelGT = mini_batch['Scene Index'].cpu()
            gray_segmentation_tensor=process_segmentation_output_cpu(semantic_scores, gray_images_tensor, batch_size=len(gray_images_np), threshold=0.5)

            imgNscore = [RGB_image, gray_segmentation_tensor] 
            RGB_image.cpu()
            gray_segmentation_tensor.cpu()
            model_1.cpu()     
            outputSceneLabel, _, _, _ = model_1.forward_sascene(imgNscore)
            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = model_1.loss(outputSceneLabel, sceneLabelGT)

            # Measure Top1, Top2 and Top5 accuracy
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

            # Update values
            losses.update(loss.item(), batch_size)
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

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(2, 3, 32, 32)):
    model_1.to(device)
    model_2.to(device)
    
    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        x = x.cpu()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print('Model equivalence test sample failed: ')
            print(y1)
            print(y2)
            return False
        
    return True


def train(dataloader, model_1, optimizer, scheduler= None, device = 'cuda'):
    best_val_acc, best_val_loss = 0, 100

    best_model = None
    model_1.to('cuda')
    model2.to('cuda')
    model_1.train()
    model2.eval()

    patient = 0
    for epoch in range(1, 101):
    # for epoch in range(1, CONFIG['TRAINING']['EPOCHS']+1):
        train_loss = []
        for i, (mini_batch) in tqdm(enumerate(dataloader)):
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                sceneLabelGT = mini_batch['Scene Index'].cuda()

                RGB_images_np = RGB_image.cpu().numpy().transpose(0, 2, 3, 1)
                gray_images_np = rgb_to_gray(RGB_images_np)
                gray_images_tensor = torch.tensor(gray_images_np).unsqueeze(1).cuda()
                
                semantic_scores = model2(RGB_image, feed_dict=None, segSize=(224,224))
                gray_segmentation_tensor=process_segmentation_output_cuda(semantic_scores, gray_images_tensor, batch_size=len(gray_images_np), threshold=0.5)

            optimizer.zero_grad()

            imgNscore = [RGB_image, gray_segmentation_tensor]
            outputSceneLabel, _, _, _ = model_1.forward_sascene(imgNscore)
            # outputSceneLabel, _, _, _ = model_1(imgNscore)
            
            loss = model_1.loss(outputSceneLabel, sceneLabelGT)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_top1, _val_top2, _val_top5, _val_loss, _ = validation_cuda(val_loader, model_1, set='Validation')
        _train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_top1:.5f} and {_val_top2:.5f} and {_val_top5:.5f}]')
        print(f'\t current patience: {patient}')

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
            # torch.save(best_model.state_dict(), f'best_model_test_epoch_{epoch}.pt')
            patient = 0
        else:
            patient+=1
        if patient == 20:
           break

    return best_model


os.environ["CUDA_VISIBLE_DEVICE"]="1"
cudnn.benchmark = torch.cuda.is_available()

CONFIG = yaml.safe_load(open("Config/config_MITIndoor_365_finetuning_qat.yaml", 'r'))



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


net_encoder = ModelBuilder.build_encoder(
    arch="resnet50",
    fc_dim=2048,
    weights="ckpt/ade20k-resnet50-upernet/encoder_epoch_30.pth")
net_decoder = ModelBuilder.build_decoder(
    arch="upernet",
    fc_dim=2048,
    num_class=150,
    weights="ckpt/ade20k-resnet50-upernet/decoder_epoch_30.pth",
    use_softmax=True)
crit = nn.NLLLoss(ignore_index=-1)

model2 = SegmentationModule(net_encoder, net_decoder, crit)

completePath = CONFIG['MODEL']['PATH'] + CONFIG['MODEL']['NAME'] + '.pth'
model = SASceneNet('ResNet-18', completePath, scene_classes=6, semantic_classes=149)

if os.path.isfile(completePath):
    print("Loading model {} from path {}...".format(CONFIG['MODEL']['NAME'], completePath))
    checkpoint = torch.load(completePath)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('*'*89)
    print(f'process for Model Quantization starts: 1. bring the PRE-TRAINED model')
    print(f"Loaded model {CONFIG['MODEL']['NAME']} from path {completePath}.")
    print(f"     Epochs {checkpoint['epoch']}")
    print('*'*89)
else:
    print(f'No checkpoint found. Train process starts without pretrained model')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    model.cuda()
cudnn.benchmark = USE_CUDA
model.eval()

quantized_model1 = QuantizableResnet18(arch='ResNet-18', completePath=completePath, scene_classes=6, semantic_classes=149)

quantized_model1.cpu().train()
quantized_model1.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.backends.quantized.engine = 'fbgemm'
torch.quantization.quantize_dynamic()

quantized_model1.eval()
quantized_model1.fuse_model()
model_parameters = filter(lambda p: p.requires_grad, quantized_model1.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of params: {}'. format(params))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('-' * 65)


optimizer = torch.optim.Adagrad(model.parameters(), lr=CONFIG['TRAINING']['LR'],weight_decay=CONFIG['TRAINING']['WEIGHT_DECAY'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)


'''train or load'''
'''train'''
model.to('cuda').train()
model = train(train_loader, model, optimizer, scheduler= None, device=device)


'''load'''
'''quantized_model1.to('cuda').train()
quantized_model1 = train(train_loader, quantized_model1, optimizer, scheduler= None, device=device)

completePath_qat = './best_model_test_epoch_1.pth' 
checkpoint_qat = torch.load(completePath_qat)
quantized_model1 = torch.quantization.prepare(quantized_model1)
quantized_model1.load_state_dict(checkpoint_qat['state_dict'], strict=False) 

quantized_model1 = quantized_model1.to('cpu')
quantized_model1 = torch.quantization.convert(quantized_model1, inplace=True)
'''
val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = validation_cuda(val_loader, model, set='Validation')
# val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = validation_cpu(val_loader, quantized_model1, set='Validation')

print('-' * 65)
print('Evaluation statistics:')

print('Validation results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
      'Mean Class Accuracy {MCA:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5))
print("complete")
