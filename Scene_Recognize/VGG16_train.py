import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from SASceneNet import SASceneNet
from Libs.Datasets.CustomDataset import CustomDataset
from Libs.Utils import utils
from Libs.Utils.torchsummary import torchsummary
import numpy as np
import yaml
from tqdm import tqdm
import random
from mit_semseg.models import ModelBuilder, SegmentationModule
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import VGG16_Weights

global USE_CUDA, classes, CONFIG
best_val_loss_list = []
best_test_loss_list = []
best_val_acc_list = []
best_test_acc_list = []

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42) # Seed 고정

def process_segmentation_output(segmentation_output, gray_images, batch_size, threshold=0.3):
    # 확률 임계값보다 높은 부분만 남기고, 나머지는 0으로 설정
    thresholded_output = torch.where(segmentation_output > threshold, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())

    # 흑백 이미지에 채널 차원 추가 (10, 1, 224, 224)
    # segmentation_output의 차원에 맞게 gray_images_with_channel 차원 조정
    gray_images_with_channel = gray_images.expand(batch_size, 150, 224, 224)

    # 흑백 이미지와 임계값이 적용된 segmentation 출력을 곱함
    final_output = gray_images_with_channel * thresholded_output
    
    return final_output

def train(dataloader, model, optimizer, scheduler= None, device = 'cuda'):
    # Start data time
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    
    best_val_acc = 0
    best_val_loss = 100
    best_test_loss = 100

    best_model = None
    model.to(device)
    patient = 0
    for epoch in range(1, CONFIG['TRAINING']['EPOCHS']+1):
    
        model.train()
        train_loss = []
        for i, (mini_batch) in tqdm(enumerate(dataloader)):
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                sceneLabelGT = mini_batch['Scene Index'].cuda()

            #TODO: make_one_hot 수정
            optimizer.zero_grad()

            outputSceneLabel = model(RGB_image)
            
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]
            prec1, prec2, prec5 = utils.accuracy(outputSceneLabel.data, sceneLabelGT, topk=(1, 2, 5))

            loss = loss_fn(outputSceneLabel, sceneLabelGT)

            loss.backward()
            losses.update(loss.item(), len(RGB_image))
            top1.update(prec1.item(), len(RGB_image))
            top2.update(prec2.item(), len(RGB_image))
            top5.update(prec5.item(), len(RGB_image))

            optimizer.step()

            train_loss.append(loss.item())

            #validation: _val_top1, val_loss만 사용하기
        _val_top1, _, _, _val_loss, _ = validation(val_loader, model, set='Validation')
        _test_top1, _, _, _test_loss, _ = inference(test_loader, model, set='Test')

        _train_loss = np.mean(train_loss)

        val_acc_list.append(_val_top1)
        val_loss_list.append(_val_loss)
        test_acc_list.append(_test_top1)
        test_loss_list.append(_test_loss)
        train_acc_list.append(top1.avg)
        train_loss_list.append(_train_loss)


        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Train ACC : [{top1.avg:.5f}]')
        print(f'Epoch [{epoch}], Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_top1:.5f}]')
        print(f'Epoch [{epoch}], Test Loss : [{_test_loss:.5f}] Test ACC : [{_test_top1:.5f}]')

        #if scheduler is not None:
        #    scheduler.step(_val_top1)
        #    print("Adjust step")

        if best_val_loss > _val_loss:
            best_val_acc = _val_top1
            best_val_loss = _val_loss
            best_model = {
                'state_dict': model.state_dict(),
                'arch': model.__class__.__name__,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            patient = 0
        else:
            patient+=1

        if patient == 20:
            break

        if best_test_loss > _test_loss:
            best_test_acc = _test_top1
            best_test_loss = _test_loss
            best_model = {
                'state_dict': model.state_dict(),
                'arch': model.__class__.__name__,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
        print(f'Patient: {patient}')
        with open(f"output/kfold_VGG16_colort/{label}/output.txt", "a") as f:
            f.write(f'Epoch {epoch} Train Loss {_train_loss:.5f} Train ACC {top1.avg:.5f}\n')
            f.write(f'Epoch {epoch} Val Loss {_val_loss:.5f} Val ACC {_val_top1:.5f}\n')
            f.write(f'Epoch {epoch} Test Loss {_test_loss:.5f} Test ACC {_test_top1:.5f}\n')

    return best_model

def validation(dataloader, model, set):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter() 
    top5 = utils.AverageMeter()

    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()

    model.eval()

    # Extract batch size
    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['VAL']

    # Start data time
    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                sceneLabelGT = mini_batch['Scene Index'].cuda()

            #TODO: make_one_hot 수정
            optimizer.zero_grad()

            outputSceneLabel = model(RGB_image)
            
            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = loss_fn(outputSceneLabel, sceneLabelGT)

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


        return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic

def inference(dataloader, model, set):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top2 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    ClassTPs_Top1 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top2 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    ClassTPs_Top5 = torch.zeros(1, len(classes), dtype=torch.uint8).cuda()
    
    # Extract batch size
    batch_size = CONFIG['TEST']['BATCH_SIZE']['TEST']

    # Start data time
    model.to(device)

    model.eval()
    
    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            start_time = time.time()
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                sceneLabelGT = mini_batch['Scene Index'].cuda()

            #TODO: make_one_hot 수정
            optimizer.zero_grad()

            outputSceneLabel = model(RGB_image)
            
            # Compute class accuracy
            ClassTPs = utils.getclassAccuracy(outputSceneLabel, sceneLabelGT, len(classes), topk=(1, 2, 5))
            ClassTPs_Top1 += ClassTPs[0]
            ClassTPs_Top2 += ClassTPs[1]
            ClassTPs_Top5 += ClassTPs[2]

            # Compute Loss
            loss = loss_fn(outputSceneLabel, sceneLabelGT)

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
  
        return top1.avg, top2.avg, top5.avg, losses.avg, ClassTPDic

def initial_setting():
    global CONFIG
    os.environ["CUDA_VISIBLE_DEVICE"]="1"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Decode CONFIG file information
    CONFIG = yaml.safe_load(open("Config/config_data_kfold.yaml", 'r'))
    print('-' * 65)
    print("Evaluation starting...")
    print('-' * 65)

    print('Evaluating complete model')
    print('Selected RG backbone architecture: ' + CONFIG['MODEL']['ARCH'])


    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # VGG-16의 마지막 fc layer를 6개 출력 노드로 수정
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # 마지막 layer를 제외한 모든 layers
    features.extend([nn.Linear(num_features, 6)]) # 마지막 layer에 6개의 출력 노드를 가진 새로운 fc layer 추가
    model.classifier = nn.Sequential(*features) # 수정된 classifier로 VGG-16 모델 업데이트

    USE_CUDA = torch.cuda.is_available()

    # Move Model to GPU an set it to evaluation mode
    if USE_CUDA:
        model.cuda()
    cudnn.benchmark = USE_CUDA
    model.eval()

    print('-' * 65)
    print('Loading dataset {}...'.format(CONFIG['DATASET']['NAME']))
    return model, device, USE_CUDA

#initial_dataloader에서 label 정의후 case 쭉 돌리기.
def initial_dataloader(label): 
    #데이터 경로
    # \Data\Datasets\Data_Case\1
    datadir = CONFIG['DATASET']['ROOT']

    train_dataset = CustomDataset(datadir, str(label) + "/train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
                                                shuffle=False, num_workers=0, pin_memory=True)

    val_dataset = CustomDataset(datadir, str(label) + "/val", tencrops=CONFIG['VALIDATION']['TEN_CROPS'], SemRGB=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], shuffle=False,
                                                num_workers=0, pin_memory=True)
    
    test_dataset = CustomDataset(datadir, str(label) + "/test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['TEST']['BATCH_SIZE']['TEST'], shuffle=False,
                                               num_workers=0,pin_memory=True)
    
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader

train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
test_acc_list = []
test_loss_list = []
model, device, USE_CUDA = initial_setting()
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = initial_dataloader(label)

data_root = CONFIG['DATASET']['ROOT']

classes = train_dataset.classes

# Print dataset information
print('Dataset loaded!')
print('Dataset Information:')
print('Train set. Size {}. Batch size {}. Nbatches {}'
    .format(len(train_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'], len(train_loader)))
print('Validation set. Size {}. Batch size {}. Nbatches {}'
    .format(len(val_loader) * CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], CONFIG['VALIDATION']['BATCH_SIZE']['VAL'], len(val_loader)))
print('Test set. Size {}. Batch size {}. Nbatches {}'
    .format(len(test_loader) * CONFIG['TEST']['BATCH_SIZE']['TEST'], CONFIG['TEST']['BATCH_SIZE']['TEST'], len(val_loader)))
print('Train set number of scenes: {}' .format(len(classes)))
print('Validation set number of scenes: {}' .format(len(classes)))
print('Test set number of scenes: {}' .format(len(classes)))

print('-' * 65)

print('Computing histogram of scene classes...')

# Check if OUTPUT_DIR exists and if not create it
if not os.path.exists(CONFIG['EXP']['OUTPUT_DIR']):
    os.makedirs(CONFIG['EXP']['OUTPUT_DIR'])

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
torchsummary.summary(model, [(3, 224, 224)], batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TRAIN'])

print('Evaluating dataset ...')

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = CONFIG['TRAINING']['WEIGHT_DECAY'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
infer_model = train(train_loader, model, optimizer, scheduler= None, device=device)

#graph
plt.clf()
plt.plot(np.arange(1,len(val_acc_list)+1), train_acc_list, label = 'train_acc')
plt.plot(np.arange(1,len(val_acc_list)+1), val_acc_list, label = 'val_acc')
plt.plot(np.arange(1,len(val_acc_list)+1), test_acc_list, label = 'test_acc')
plt.legend()
plt.savefig(f'./output/acc_graph.png')
plt.clf()

plt.plot(np.arange(1,len(val_acc_list)+1), train_loss_list, label = 'train_loss')
plt.plot(np.arange(1,len(val_acc_list)+1), val_loss_list, label = 'val_loss')
plt.plot(np.arange(1,len(val_acc_list)+1), test_loss_list, label = 'test_loss')
plt.legend()
plt.savefig(f'./output/loss_graph.png')
plt.clf()

# Evaluate model on validation set
val_top1, val_top2, val_top5, val_loss, val_ClassTPDic = validation(val_loader, model, set='Validation')
test_top1, test_top2, test_top5, test_loss, test_ClassTPDic = inference(test_loader, model, set='Test')

with open(f"output/output.txt", "a") as f:
    f.write(f'\n')
    f.write(f'Best model: Val Loss {val_loss:.5f} Val ACC {val_top1:.5f}\n')
    f.write(f'Best model: Test Loss {test_loss:.5f} Test ACC {test_top1:.5f}\n')

# Print complete evaluation information
print('-' * 65)
print('Evaluation statistics:')


print('Validation results: Loss {val_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}'.format(val_loss=val_loss, top1=val_top1, top2=val_top2, top5=val_top5))
print('Test results: Loss {test_loss:.3f}, Prec@1 {top1:.3f}, Prec@2 {top2:.3f}, Prec@5 {top5:.3f}'.format(test_loss=test_loss, top1=test_top1, top2=test_top2, top5=test_top5))
torch.save(infer_model, f'./output/model.pth')
