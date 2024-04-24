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
from train import test_loader
from train import loss_function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_iou_array = [];

def test2():
    model = torch.load('model/unetc_model.pth',map_location=torch.device(device))
    model.eval()
    iou_metric = JaccardIndex(task='binary').to(device)
    total_time = 0
    running_val_fps=0
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            #print("hii",images.shape[0])
            torch.cuda.synchronize(device)
            start_time = time.time()  # Start time for measuring fps
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1)
            outputs = model(images)
            outputs=torch.sigmoid(outputs)
            loss = loss_function(outputs, masks)
            pred_masks = torch.argmax(outputs, dim=1)
            #print("masks",masks)
            iou_metric(outputs, masks)
            torch.cuda.synchronize(device)
            end_time = time.time()  # End time for measuring fps
            time_taken = time.time() - start_time
            total_time += end_time - start_time
            running_val_fps += images.shape[0] / time_taken
    
    
    # Calculate frames per second (fps)
    total_images = len(test_loader.dataset)
    avg_time_per_image = total_time / total_images
    fps = 1 / avg_time_per_image
    fps2 = running_val_fps / len(test_loader)
    miou = iou_metric.compute()
    print("Test IOU is: ",miou.item()*100)
    print("FPS is: ",fps)
    print("FPS2 is: ",fps2)
    return miou, fps

if __name__ == "__main__":
    test2()