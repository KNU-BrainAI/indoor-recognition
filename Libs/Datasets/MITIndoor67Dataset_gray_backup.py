from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch
from imgaug import augmenters as iaa
import cv2

class MITIndoor67Dataset(Dataset):
    """Class for MIT Indoor 67 dataset."""

    def __init__(self, root_dir, set, tencrops=True, SemRGB=True):
        """
        Initialize the dataset. Read scene categories, get number of classes, create filename and ground-truth labels
        lists, create ImAug and PyTorch transformations

        :param root_dir: Root directory to the dataset
        :param set: Dataset set: Training or Validation
        """
        # Extract main path and set (Train or Val).
        self.image_dir = root_dir
        self.set = set
        # Set boolean variable of ten crops for validation
        self.TenCrop = tencrops

        self.SemRGB = SemRGB
        if SemRGB:
            self.RGB = "_RGB"
        else:
            self.RGB = ""

        # Decode dataset scene categories
        self.classes = list()
        class_file_name = os.path.join(root_dir, "scene_names.txt")

        with open(class_file_name) as class_file:
            for line in class_file:
                line = line.split()[0]
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Check if there a class with a subclass inside (outdoor, indoor)
                if len(split_indices) > 2:
                    line = line[:split_indices[2]] + '-' + line[split_indices[2]+1:]

                self.classes.append(line[split_indices[1] + 1:])

        # Get number of classes
        self.nclasses = self.classes.__len__()

        # Create list for filenames and scene ground-truth labels
        self.filenames = list()
        self.labels = list()
        self.labelsindex = list()
        filenames_file = os.path.join(root_dir, (set + ".txt"))
        print(filenames_file)
        # Fill filenames list and ground-truth labels list
        with open(filenames_file) as class_file:
            for line in class_file:
                # if random.random() > 0.6 or (self.set is "val"):
                split_indices = [i for i, letter in enumerate(line) if letter == '/']
                # Obtain name and label
                
                if set == 'train' or set == 'val':
                    name = line[split_indices[1] + 1:-1]
                    label = line[split_indices[0] + 1: split_indices[1]]
                else:
                    name = line[split_indices[2] + 1:-1]
                    label = line[split_indices[1] + 1: split_indices[2]]

                self.filenames.append(name)
                self.labels.append(label)
                self.labelsindex.append(self.classes.index(label))

        # Control Statements for data loading
        assert len(self.filenames) == len(self.labels)

        # ----------------------------- #
        #     ImAug Transformations     #
        # ----------------------------- #
        # Transformations for train set
        self.seq = iaa.Sequential([
            # Small gaussian blur with random sigma between 0 and 0.5.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=True)  # apply augmenters in random order

        self.seq_sem = iaa.Sequential([
            iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
        ], random_order=True)

        # ----------------------------- #
        #    Pytorch Transformations    #
        # ----------------------------- #
        self.mean = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.resizeSize = 256
        self.outputSize = 224

        # Train Set Transformation
        self.train_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.STD)
            
        ])
        self.train_transforms_scores = transforms.ToTensor()

        if not SemRGB:
            self.train_transforms_sem = transforms.Lambda(
                lambda sem: torch.unsqueeze(torch.from_numpy(np.asarray(sem) + 1).long(), 0))
        else:
            self.train_transforms_sem = transforms.Lambda(
                lambda sem: torch.from_numpy(np.asarray(sem) + 1).long().permute(2, 0, 1))

        # Transformations for validation set

        self.val_transforms_img = transforms.Compose([
            transforms.Resize(self.resizeSize),
            transforms.CenterCrop(self.outputSize),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.STD)
        ])


    def __len__(self):
        """
        Function to get the size of the dataset
        :return: Size of dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Function to get a sample from the dataset. First both RGB and Semantic images are read in PIL format. Then
        transformations are applied from PIL to Numpy arrays to Tensors.

        For regular usage:
            - Images should be outputed with dimensions (3, W, H)
            - Semantic Images should be outputed with dimensions (1, W, H)

        In the case that 10-crops are used:
            - Images should be outputed with dimensions (10, 3, W, H)
            - Semantic Images should be outputed with dimensions (10, 1, W, H)

        :param idx: Index
        :return: Dictionary containing {RGB image, semantic segmentation mask, scene category index}
        """
        
        # Get RGB image path and load it
        if set == 'train' or set == 'val':
            img_name = os.path.join(self.image_dir, self.set, self.labels[idx], self.filenames[idx])
        else:
            img_name = os.path.join(self.image_dir, self.set, self.labels[idx], self.filenames[idx])
        img = Image.open(img_name)
        #흑백이미지가 아니라면 흑백으로 변환
        try:
            img = img.convert('L')
            img = img.convert('RGB')
        except:
            print("already gray-scale")
            pass
        #CV -> 
        #PIL -> 
        # RGB to gray
        
        # Convert it to RGB if gray-scale
        # 먼저 테스트 필요. or 이미지 전처리를 여기서한다면 직접해주기
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply transformations depending on the set (train, val)
        if self.set == "train":
            # Define Random crop. If image is smaller resize first.
            bilinearResize_trans = transforms.Resize(self.resizeSize)

            img = bilinearResize_trans(img)

            # Extract Random Crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.outputSize, self.outputSize))
            # Apply Random Crop parameters
            img = TF.crop(img, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)

            # Apply transformations from ImgAug library
            img = np.asarray(img)
            img = np.squeeze(self.seq.augment_images(np.expand_dims(img, axis=0)))

            # Apply not random transforms. To tensor and normalization for RGB. To tensor for semantic segmentation.
            img = self.train_transforms_img(img)

        else:
            img = self.val_transforms_img(img)

        img_copy = img.clone()
        # Create dictionary
        self.sample = {'Image': img_copy, 'Scene Index': self.classes.index(self.labels[idx])}

        return self.sample