"""
Evalaution file to completly val the trained model
Usage:
    --ConfigPath [PATH to configuration file for desired dataset]
"""
import os
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from SASceneNet import SASceneNet
from Libs.Datasets.MITIndoor67Dataset_gray import MITIndoor67Dataset
from Libs.Utils import utils
import numpy as np
import yaml
import random
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from mit_semseg.models import ModelBuilder, SegmentationModule
import cv2

os.environ["CUDA_VISIBLE_DEVICE"]="1"

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

prediction_save = []
targets = []

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42) # Seed 고정

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

def inference(dataloader, model, set):
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
    batch_size = CONFIG['TEST']['BATCH_SIZE']['TEST']

    # Start data time
    data_time_start = time.time()
    model.to(device)
    model2.to(device)

    model.eval()
    model2.eval()
    
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
                
            # Model Forward
            outputSceneLabel, _, _, _ = model(RGB_image, gray_segmentation_tensor)

            #print("output Scene label : ")
            #print(outputSceneLabel)# prediction 레이블
            #print("scene LabelGT : ")
            #print(sceneLabelGT) #정답 레이블
            for pred_labels in outputSceneLabel:
                prediction_save.append(torch.nonzero(pred_labels == max(pred_labels)))
            #concatenation 필요
            
            targets.append(sceneLabelGT)

            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = model.loss(outputSceneLabel, sceneLabelGT)

            # Measure Top1, Top2 and Top5 accuracy
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))
          
            """
            # prediciton top 1 저장 total
            print("-----------------------------")
            print(outputSceneLabel)
            print(sceneLabelGT)
            _, predicted = outputSceneLabel.max(1)
            print(predicted)
            print("-----------------------------")
            """
            # Update values
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top2.update(prec2.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # Measure batch elapsed time
            batch_time.update(time.time() - start_time)

            # Print information
            if i % CONFIG['TEST']['PRINT_FREQ'] == 0:
                print('valing {} set batch: [{}/{}]\t'
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

        #TODO: precision값들 정리하는게 필요
        #original_scene = [8,11,8,11,10,10,10,12,11,10]
    
        return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic

# Decode CONFIG file information
CONFIG = yaml.safe_load(open("Config/config_MITIndoor_test.yaml", 'r'))
USE_CUDA = torch.cuda.is_available()

print('-' * 65)
print("Evaluation starting...")
print('-' * 65)

print('Evaluating complete model')
print('Selected RG backbone architecture: ' + CONFIG['MODEL']['ARCH'])

# Load the trained model
model = SASceneNet(arch='ResNet-18', scene_classes=6, semantic_classes=149)

# Load the trained model
#completePath = CONFIG['MODEL']['PATH'] + 'TRAINSAVE/' + 'best_model_epoch.pth'
completePath = './best_model_gray_epoch_15.pth'
load_model = torch.load(completePath)
model.load_state_dict(load_model['state_dict'])
print()

model.to(device)
model2.to(device)

model.eval()
model2.eval()

# Move Model to GPU an set it to evaluation mode
if USE_CUDA:
    model.cuda()
cudnn.benchmark = USE_CUDA

print('-' * 65)
print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))


test_dir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])
test_dataset = MITIndoor67Dataset(test_dir, "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['TEST']['BATCH_SIZE']['TRAIN'],
                                            shuffle=False, num_workers=0, drop_last=True,pin_memory=True)

classes = test_dataset.classes
print(f"class : {classes}")
# Print dataset information
print('Dataset loaded!')
print('Dataset Information:')
print('val set. Size {}. Batch size {}. Nbatches {}'
      .format(len(test_loader) * CONFIG['TEST']['BATCH_SIZE']['TEST'], CONFIG['TEST']['BATCH_SIZE']['TEST'], len(test_loader)))
print('Train set number of scenes: {}' .format(len(classes)))
print('VAL set number of scenes: {}' .format(len(classes)))

print('-' * 65)

print('Computing histogram of scene classes...')

ValHist = utils.getHistogramOfClasses(test_loader, classes, "val")

# Check if OUTPUT_DIR exists and if not create it
if not os.path.exists(CONFIG['EXP']['OUTPUT_DIR']):
    os.makedirs(CONFIG['EXP']['OUTPUT_DIR'])

# Save Dataset histograms
#np.savetxt(CONFIG['EXP']['OUTPUT_DIR'] + '/valHist.txt', ValHist, '%u')

# Print Network information
print('-' * 65)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of params: {}'. format(params))
print('-' * 65)
print('GPU in use: {} with {} memory'.format(torch.cuda.get_device_name(0), torch.cuda.max_memory_allocated(0)))
print('-' * 65)

# Summary of the network for a dummy input
sample = next(iter(test_loader))
print(sample['Image'].size()) # 10,10,3,224,224
#torchsummary.summary(model, [(3, 224, 224), (CONFIG['DATASET']['N_CLASSES_SEM'] + 1, 224, 224)], batch_size=CONFIG['val']['BATCH_SIZE']['TRAIN'])

print('Evaluating dataset ...')

# Evaluate model on inference set
val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = inference(test_loader, model, set='val')

prediction_save = torch.cat(prediction_save, dim = 0)
prediction_save = prediction_save.squeeze()
targets = torch.cat(targets, dim = 0)

prediction_save=prediction_save.to(device='cpu')
targets=targets.to(device='cpu')

prediction_np = prediction_save.numpy()
targets_np = targets.numpy()

print(f'prediction_save: {prediction_save}')
print(f'targets: {targets}')
class_names = classes

#confusion matrix
cm = confusion_matrix(targets, prediction_save)
for cm_l in cm:
    for cm_n in cm_l:
        print(f"{cm_n}\t",end="")
    print("\n",end="")
for i, class_1 in enumerate(class_names):
    class_targets = (targets == i).long()
    class_preds = (prediction_save == i).long()

    acc = accuracy_score(class_targets, class_preds)
    precisions = precision_score(class_targets, class_preds)
    recalls = recall_score(class_targets, class_preds)
    f1_scores = f1_score(class_targets, class_preds)

    print(f'{class_1} - Accuracy: {acc:.2f}, Precision: {precisions:.2f}, Recall: {recalls:.2f}, F1-score: {f1_scores:.2f}')

# show confusion matrix
# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xticks(np.arange(6), class_names, rotation= 45)
plt.yticks(np.arange(6), class_names, rotation= 0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
#plt.show()

# Print complete evaluation information
print('-' * 65)
print('Evaluation statistics:')

print('val results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}, '
      'Mean Class Accuracy {MCA:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5,
                                              MCA=1))