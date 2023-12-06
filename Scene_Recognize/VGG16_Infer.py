"""
완료
"""


import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from Libs.Utils import utils
import numpy as np
import yaml
import torchvision.transforms as transforms
import time
from PIL import Image
from torchvision import models
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICE"]="1"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

root = r"./Data/Datasets/Data_Case/7/4/test/grocerystore"
img_path = r"./imageroot기입"

root = r"./Data/Datasets/Data_Case/3/test/class_513"
model_path = r'./output/kfold_VGG16_AUG/4/model_case_4_val.pth'

model = models.vgg16().to(device)
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # 마지막 layer를 제외한 모든 layers
features.extend([nn.Linear(num_features, 6)]) # 마지막 layer에 6개의 출력 노드를 가진 새로운 fc layer 추가
model.classifier = nn.Sequential(*features) # 수정된 classifier로 VGG-16 모델 업데이트

load_model = torch.load(model_path)
model.load_state_dict(load_model['state_dict'])

model.cuda()
model.eval()
# Load the trained model


mean = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# 이미지 변환
transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, STD)
        ])
pred_list = []

def classification(image):
    # model 추론
    image = image.unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)

    return predicted


def undistort(image, balance=0.2, dim2=None, dim3=None):
    #광각카메라 보정
    DIM=(640, 480)
    K=np.array([[311.97650208828185, 0.0, 320.6395514817178], [0.0, 311.11476247716445, 240.8457049772965], [0.0, 0.0, 1.0]])
    D=np.array([[-0.0029847411613313384], [-0.09710750673858538], [0.07259558950520098], [-0.023376998019068302]])

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
    img_path = os.path.join(root, filename)
    try:
        image = Image.open(img_path).convert('L').convert('RGB')
        image = np.array(image)
        image = undistort(image)
        image = transform(image).to(device)
    except:
        print("Image Transform Error")
        break

    with open('mit_names.txt', 'r') as file:
        classes = [line.split(' ')[0] for line in file]

    predicted = classification(image)
    
    pred_list.append(classes[predicted.item()])
    print("이미지 분류 결과:", classes[predicted.item()], end=' ')
    print(f"이미지 파일: {img_path}")
