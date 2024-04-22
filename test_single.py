import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToPILImage
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms 
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import random_split
from PIL import Image
import cv2
import albumentations as A
import time
import os
from tqdm import tqdm
import random
from torchsummary import summary
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
from data import CustomDataset
from preprocess import u_transform, t_train, t_test, t_val
from metrics import Iou_score
from unet import test_loader
from unet import loss_function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_iou_array = [];
test_loss_arrray = [];

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((256, 256)),
                                transforms.Normalize(mean, std)])
def test():
    start_time = time.time()
    model = torch.load('model/model_unet_bce.pth')
    model.eval()
    image, mask = Image.open('oo.jpg') , Image.open('dataset/Images/image_0.jpg') 
    image=transform(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():
        start_time = time.time()
        image = image.unsqueeze(0)
        output = model(image)
        output=torch.sigmoid(output)
        # threshold = 0.5
        # binary_predictions = (output > threshold).float()
        threshold = 0.5  # Adjust this threshold as needed
        binary_mask = (output > threshold).float()
        # Visualize the output mask using Matplotlib
        binary_mask_np = binary_mask.squeeze().cpu().numpy()

        # Visualize the binary mask
        plt.imshow(binary_mask_np, cmap='gray')
        plt.axis('off')
        plt.show()
        # torch.set_printoptions(profile="full")
        # #print("outout is",output)
        # masked = torch.argmax(output, dim=1)
        # print("masked is>>",masked)
        # masked = masked.cpu().squeeze(0)
        # end_time = time.time()  # End time for measuring fps
        # total_time = end_time - start_time
        # print("total time is",total_time)
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
        # ax1.imshow(masked)
        # plt.show()  

if __name__ == "__main__":
    test()