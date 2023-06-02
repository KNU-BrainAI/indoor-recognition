"""
Evalaution file to completly test the trained model
Usage:
    --ConfigPath [PATH to configuration file for desired dataset]
"""
import os
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from SASceneNet import SASceneNet
from Libs.Datasets.MITIndoor67Dataset_gray import MITIndoor67Dataset
from Libs.Utils import utils
from Libs.Utils.torchsummary import torchsummary
import numpy as np
from tqdm import tqdm
import random
from mit_semseg.models import ModelBuilder, SegmentationModule
import cv2
import yaml
os.environ["CUDA_VISIBLE_DEVICE"]="1"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#이것도 모델안에 넣어두는게 best일듯
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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42) # Seed 고정

def validation(dataloader, model, set):
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

    # Extract batch size
    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['VAL']

    # Start data time
    data_time_start = time.time()

    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                RGB_images_np = RGB_image.cpu().numpy().transpose(0, 2, 3, 1)
                gray_images_np = rgb_to_gray(RGB_images_np)
                #gray_images_np shape (10,224,224)
                gray_images_tensor = torch.tensor(gray_images_np).unsqueeze(1).cuda()
                
                semantic_scores = model2(RGB_image, feed_dict=None, segSize=(224,224))
                #print(semantic_scores.shape) [10,150,224,224]
                #여기서 10,3,224,224가 되야한다.
                # 아마 ade20k 기준으로 한다면 10, 150 ,224,224가 나올듯
                # model output 값을 ordered dict에서 tensor로 변환도 필요. -> key 'out'일때 output tensor
                sceneLabelGT = mini_batch['Scene Index'].cuda()
                gray_segmentation_tensor=process_segmentation_output(semantic_scores, gray_images_tensor, threshold=0.5, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
                
                sceneLabelGT = mini_batch['Scene Index'].cuda()

            # Model Forward
            outputSceneLabel, _, _, _ = model(RGB_image, gray_segmentation_tensor)
            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = model.loss(outputSceneLabel, sceneLabelGT)

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

#batch size안에 있는 image를 하나씩 converting해서 stack으로 채움.
def rgb_to_gray(images):
    gray_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray_image)
    return np.stack(gray_images)

def process_segmentation_output(segmentation_output, gray_images, threshold=0.5, batch_size=10):
    # 확률 임계값보다 높은 부분만 남기고, 나머지는 0으로 설정
    thresholded_output = torch.where(segmentation_output > threshold, segmentation_output, torch.tensor(0.0).cuda())

    # 흑백 이미지에 채널 차원 추가 (10, 1, 224, 224)
    # segmentation_output의 차원에 맞게 gray_images_with_channel 차원 조정
    gray_images_with_channel = gray_images.expand(batch_size, 150, 224, 224)

    # 흑백 이미지와 임계값이 적용된 segmentation 출력을 곱함
    final_output = gray_images_with_channel * thresholded_output

    return final_output

def train(dataloader, model, optimizer, scheduler= None, device = 'cuda'):
    # Start data time
    best_val_acc = 0
    best_val_loss = 100

    best_model = None
    model.to(device)
    model2.to(device)
    patient = 0
    for epoch in range(1, CONFIG['TRAINING']['EPOCHS']+1):
        model.train()
        train_loss = []
        print(f'train epoch : {epoch}, running')
        for i, (mini_batch) in tqdm(enumerate(dataloader)):
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                RGB_images_np = RGB_image.cpu().numpy().transpose(0, 2, 3, 1)
                gray_images_np = rgb_to_gray(RGB_images_np)
                #gray_images_np shape (10,224,224)
                gray_images_tensor = torch.tensor(gray_images_np).unsqueeze(1).cuda()
                
                semantic_scores = model2(RGB_image, feed_dict=None, segSize=(224,224))
                #print(semantic_scores.shape) [10,150,224,224]
                #여기서 10,3,224,224가 되야한다.
                # 아마 ade20k 기준으로 한다면 10, 150 ,224,224가 나올듯
                # model output 값을 ordered dict에서 tensor로 변환도 필요. -> key 'out'일때 output tensor
                sceneLabelGT = mini_batch['Scene Index'].cuda()
                gray_segmentation_tensor=process_segmentation_output(semantic_scores, gray_images_tensor, threshold=0.5)

            #TODO: make_one_hot 수정
            #semanticTensor = utils.make_one_hot(semantic_mask, semantic_scores, C=CONFIG['DATASET']['N_CLASSES_SEM'])
            #print(semanticTensor.shape)
            #10,152,224,224 shape은 같다. 이제 make_one_hot 분석후, 비슷하게 바꿔야한다.
            optimizer.zero_grad()

            #outputSceneLabel, _, _, _= model(RGB_image, semanticTensor)
            outputSceneLabel, _, _, _ = model(RGB_image, gray_segmentation_tensor)
            

            loss = model.loss(outputSceneLabel, sceneLabelGT)

            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

            #validation: _val_top1, val_loss만 사용하기
        _val_top1, val_top2, val_top5, _val_loss, val_ClassTPDic = validation(val_loader, model, set='Validation')
        _train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_top1:.5f}]')

        #scheduler: step?

        if scheduler is not None:
            scheduler.step(_val_top1)
        if _val_loss < best_val_loss:
            if best_val_acc <= _val_top1:
                best_val_acc = _val_top1
                best_val_loss = _val_loss
                best_model = {
                    'state_dict': model.state_dict(),
                    'arch': model.__class__.__name__,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(best_model, f'best_model_gray_epoch_{epoch}.pth')
                patient = 0
            else:
                patient+=1
        else:
            patient+=1
        #if patient == 20:
        #    break

    return best_model

global USE_CUDA, classes, CONFIG

# Decode CONFIG file information
CONFIG = yaml.safe_load(open("Config/config_MITIndoor_365_finetuning.yaml", 'r'))
USE_CUDA = torch.cuda.is_available()

PATH = CONFIG['MODEL']['PATH'] + 'TRAINSAVE/'

print('-' * 65)
print("Evaluation starting...")
print('-' * 65)

print('Evaluating complete model')
print('Selected RG backbone architecture: ' + CONFIG['MODEL']['ARCH'])
model = SASceneNet(arch=CONFIG['MODEL']['ARCH'], scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'], semantic_classes=CONFIG['DATASET']['N_CLASSES_SEM'])

# Load the trained model + use place365 and finetuning
#completePath = CONFIG['MODEL']['PATH'] + 'TRAINSAVE/' +'best_model_epoch.pth'
completePath = CONFIG['MODEL']['PATH'] + 'SAScene_ResNet18_Places.pth.tar'

if os.path.isfile(completePath):
    print("Loading model {} from path {}...".format(CONFIG['MODEL']['NAME'], completePath))
    checkpoint = torch.load(completePath)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    #TODO : freeze 필요
    print("Loaded model {} from path {}.".format(CONFIG['MODEL']['NAME'], completePath))
    print("     Epochs {}".format(checkpoint['epoch']))
    print("     Single crop reported precision {}".format(best_prec1))
else:
    print("No checkpoint found at '{}'. Check configuration file MODEL field".format(completePath))
    quit()
    
model.in_block_sem = nn.Sequential(
            nn.Conv2d(150, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


#for number of class change
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,6)

num_features_SEM = model.fc_SEM.in_features
model.fc_SEM = nn.Linear(num_features_SEM,6)

num_features_RGB = model.fc_RGB.in_features
model.fc_RGB = nn.Linear(num_features_RGB,6)


# Move Model to GPU an set it to evaluation mode
if USE_CUDA:
    model.cuda()
cudnn.benchmark = USE_CUDA
model.eval()

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))

traindir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])
valdir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])

train_dataset = MITIndoor67Dataset(traindir, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
                                            shuffle=False, num_workers=0, pin_memory=True)

val_dataset = MITIndoor67Dataset(valdir, "val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'], SemRGB=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], shuffle=False,
                                            num_workers=0, pin_memory=True)

classes = train_dataset.classes

# Print dataset information
print('Dataset loaded!')
print('Dataset Information:')
print('Train set. Size {}. Batch size {}. Nbatches {}'
      .format(len(train_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], len(train_loader)))
print('Validation set. Size {}. Batch size {}. Nbatches {}'
      .format(len(val_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], len(val_loader)))
print('Train set number of scenes: {}' .format(len(classes)))
print('Validation set number of scenes: {}' .format(len(classes)))

print('-' * 65)

print('Computing histogram of scene classes...')

TrainHist = utils.getHistogramOfClasses(train_loader, classes, "Training")
ValHist = utils.getHistogramOfClasses(val_loader, classes, "Validation")

# Check if OUTPUT_DIR exists and if not create it
if not os.path.exists(CONFIG['EXP']['OUTPUT_DIR']):
    os.makedirs(CONFIG['EXP']['OUTPUT_DIR'])

# Save Dataset histograms
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/TrainingHist.txt', TrainHist, '%u')
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationHist.txt', ValHist, '%u')

# Print Network information
print('-' * 65)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of params: {}'. format(params))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('-' * 65)

# Summary of the network for a dummy input
sample = next(iter(val_loader))
#print(sample['Image'].size()) # 10,10,3,224,224
#torchsummary.summary(model, [(3, 224, 224), (CONFIG['DATASET']['N_CLASSES_SEM'] + 1, 224, 224)], batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'])
torchsummary.summary(model, [(3, 224, 224), (150, 224, 224)], batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'])

print('Evaluating dataset ...')


#optimizer = DFW(model.parameters(), eta=CONFIG['TRAINING']['LR'], momentum=CONFIG['TRAINING']['MOMENTUM'], weight_decay=CONFIG['TRAINING']['WEIGHT_DECAY'])

optimizer = torch.optim.Adagrad(model.parameters(), lr=CONFIG['TRAINING']['LR'],weight_decay=CONFIG['TRAINING']['WEIGHT_DECAY'])
#optimizer = torch.optim.Adam(model.parameters(),lr=1e-6 , eps=1e-7, weight_decay=4e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
infer_model = train(train_loader, model, optimizer, scheduler= None, device=device)

# Evaluate model on validation set
val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = validation(val_loader, model, set='Validation')

# Save Validation Class Accuracy
val_ClassAcc_top1 = (val_ClassTPDic['Top1'] / (ValHist + 0.0001)) * 100
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationTop1ClassAccuracy.txt', np.transpose(val_ClassAcc_top1), '%f')

val_ClassAcc_top2 = (val_ClassTPDic['Top2'] / (ValHist + 0.0001)) * 100
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationTop2ClassAccuracy.txt', np.transpose(val_ClassAcc_top2), '%f')

val_ClassAcc_top5 = (val_ClassTPDic['Top5'] / (ValHist + 0.0001)) * 100
np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/ValidationTop5ClassAccuracy.txt', np.transpose(val_ClassAcc_top5), '%f')

# Print complete evaluation information
print('-' * 65)
print('Evaluation statistics:')

print('Validation results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
      'Mean Class Accuracy {MCA:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5,
                                              MCA=np.mean(val_ClassAcc_top1)))

print("Save Model")
torch.save(model.state_dict(), PATH + 'btest_model.pt')

print("complete")
