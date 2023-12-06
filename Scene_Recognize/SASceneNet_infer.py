
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from SASceneNet import SASceneNet
from Libs.Utils import utils
import numpy as np
import yaml
from mit_semseg.models import ModelBuilder, SegmentationModule
import torchvision.transforms as transforms
import time
from PIL import Image
import cv2
import argparse
import shutil
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICE"]="1"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 수정 필요
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

model = SASceneNet(arch='ResNet-18', scene_classes=6, semantic_classes=149)

model.to(device)
model2.to(device)

# Load the trained model
completePath = './output/kfold_noen_val_sd_mitindoor/2/model_epoch_41_case_2_val.pth'
load_model = torch.load(completePath)
model.load_state_dict(load_model['state_dict'])

model.eval()
model2.eval()

def rgb_to_gray(images):
    gray_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray_image)
    return np.stack(gray_images)

def process_segmentation_output(segmentation_output, gray_images, batch_size, threshold=0.5):
    # 확률 임계값보다 높은 부분만 남기고, 나머지는 0으로 설정
    thresholded_output = torch.where(segmentation_output > threshold, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())

    # 흑백 이미지에 채널 차원 추가 (10, 1, 224, 224)
    # segmentation_output의 차원에 맞게 gray_images_with_channel 차원 조정
    gray_images_with_channel = gray_images.expand(batch_size, 150, 224, 224)

    # 흑백 이미지와 임계값이 적용된 segmentation 출력을 곱함
    final_output = gray_images_with_channel * thresholded_output
    
    return final_output

mean = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# 이미지 변환
transform = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, STD)
        ])
pred_list = []
time_list = []

def classification(image):
    image = image.unsqueeze(0)
    images_np = image.cpu().numpy().transpose(0, 2, 3, 1)
    gray_images_np = rgb_to_gray(images_np)

    start_time = time.time()
    semantic_scores = model2(image, feed_dict=None, segSize=(224,224))
    gray_images_tensor = torch.tensor(gray_images_np).unsqueeze(1).cuda()
    gray_segmentation_tensor=process_segmentation_output(semantic_scores, gray_images_tensor, threshold=0.5, batch_size=1)
    
    outputSceneLabel, _, _, _ = model(image, gray_segmentation_tensor)
    _, predicted = torch.max(outputSceneLabel, 1)
    print(outputSceneLabel)
    inf_time = time.time()-start_time

    time_list.append(inf_time) 
    print(f"Inference time : {time.time()-start_time:.2f}s")# 2.25s , #0.05s ~ 0.3s

    return predicted



class_label = [0,0,0,0,0,0]
# 한케이스만 테스트하면되니 root는 하나로 고정
root = r"./Data/Datasets/Data_Case/3/test/office_506"
save_root =  r"./Data/Datasets/Data_Case/3/test/class_hustar/"
#                       calibration
# office 506 : pred = [0, 0, 17, 0, 285, 0]     office          pred = [0, 0, 256, 0, 46, 0]    hospitalroom
# class 103 : pred = [246, 0, 0, 0, 0, 0]       classroom       pred = [221, 0, 0, 0, 0, 25]    classroom
# class hustar : pred = [0, 0, 239, 1, 2, 0]    hospitalroom    pred = [0, 0, 236, 6, 0, 0]     hospitalroom
# class 101 : pred = [89, 0, 3, 0, 0, 223]      restaurant      pred = [184, 0, 131, 0, 0, 0]   classroom
# class 513 : pred = [0, 0, 239, 0, 0, 0]       hospitalroom    pred = [0, 0, 238, 0, 0, 1]     hospitalroom

DIM=(640, 480)
K=np.array([[311.97650208828185, 0.0, 320.6395514817178], [0.0, 311.11476247716445, 240.8457049772965], [0.0, 0.0, 1.0]])
D=np.array([[-0.0029847411613313384], [-0.09710750673858538], [0.07259558950520098], [-0.023376998019068302]])

def undistort(image, balance=0.2, dim2=None, dim3=None):
    dim1 = image.shape[:2][::-1]
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1]
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0] 
    scaled_K[2][2] = 1.0 
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, 
   									 D,dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, 
    									np.eye(3), new_K,dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(image, map1, map2,
    				interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    
    #cv2.imshow("undistorted", undistorted_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    img = Image.fromarray(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    return img

for filename in os.listdir(root):
    try:
        img_path = os.path.join(root, filename)
        image = Image.open(img_path).convert('L').convert('RGB')

        #image 불러오기
        image = np.array(image)
        image = undistort(image)
        image = transform(image)
        image = image.cuda()
        
        with open('mit_names.txt', 'r') as file:
            classes = [line.split(' ')[0] for line in file]

        predicted = classification(image)
       
        class_label[predicted.item()]+=1
        pred_list.append(classes[predicted.item()])
        print("이미지 분류 결과:", classes[predicted.item()])
        #분류되는 결과에 따라 각각폴더로 저장
        #shutil.copy(img_path,save_root+classes[predicted.item()])
    except:
        print("Error")
        pass
# heat time인 첫번째 시간 제외
avg_time = sum(time_list[1:])/len(time_list[1:])
print(f"avg_time = {avg_time}")
print(f"pred = {class_label}")
