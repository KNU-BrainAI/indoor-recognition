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
import torchvision.transforms as transforms
import time
from PIL import Image
import cv2
from torchvision import models

os.environ["cpu_VISIBLE_DEVICE"]="1"

device = torch.device("cpu")


# 수정 필요
path = r'./output/kfold_VGG16/6/model_epoch_26_case_6_val.pth'
model = models.vgg16()
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # 마지막 layer를 제외한 모든 layers
features.extend([nn.Linear(num_features, 6)]) # 마지막 layer에 6개의 출력 노드를 가진 새로운 fc layer 추가
model.classifier = nn.Sequential(*features) # 수정된 classifier로 VGG-16 모델 업데이트

load_model = torch.load(path)
model.load_state_dict(load_model['state_dict'])
model.eval()

model.classifier[6] = torch.nn.Linear(num_features, 6)

# Load the trained model
model.eval()


def rgb_to_gray(images):
    gray_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray_image)
    return np.stack(gray_images)


mean = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# 이미지 변환
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, STD)
        ])
pred_list = []
time_list = []

def classification(image):
    image = image.unsqueeze(0)
    images_np = image.cpu().numpy().transpose(0, 2, 3, 1)
    start_time = time.time()

    output = model(image)

    _, predicted = torch.max(output, 1)
    inf_time = time.time()-start_time

    time_list.append(inf_time)
    print(f"Inference time : {time.time()-start_time:.2f}s")# 2.25s , #0.05s ~ 0.3s

    return predicted


# 한케이스만 테스트하면되니 root는 하나로 고정
root = r"./Data/Datasets/Data_Case/3/test/camera/office"

for filename in os.listdir(root):
    img_path = os.path.join(root, filename)
    imageo = Image.open(img_path)

    #image 불러오기
    img_np = np.array(imageo)
    replicated_image = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
    image = Image.fromarray(replicated_image)
    image = image.convert('L').convert('RGB')
    image = transform(image)
    RGB_image = image.cpu()

    print(RGB_image.shape)

    with open('mit_names.txt', 'r') as file:
        classes = [line.split(' ')[0] for line in file]

    predicted = classification(RGB_image)
    pred_list.append(classes[predicted.item()])
    print("이미지 분류 결과:", classes[predicted.item()])

# heat time인 첫번째 시간 제외
avg_time = sum(time_list[1:])/len(time_list[1:])
print(f"avg_time = {avg_time}")