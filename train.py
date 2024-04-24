import numpy as np 
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data import random_split
import albumentations as A
import os
from tqdm import tqdm
import random
import segmentation_models_pytorch as smp
from data import CustomDataset
from preprocess import u_transform, t_train, t_test, t_val
from metrics import Iou_score
from plot import plot_loss, plot_score
from unet import build_unet
torch.set_printoptions(profile="full",sci_mode=False)

lrs = []
epochs=15

test_iou = [];

train_iou_array = [];
train_loss_arrray = [];

test_iou_array = [];
test_loss_arrray = [];

val_iou_array = [];
val_loss_array = [];

batch_size=2

random_seed = 89
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = CustomDataset('dataset/Images', 'dataset/Masks',u_transform)

train_size = int(0.6 * len(full_dataset))  # 60% 
val_size = int(0.2 * len(full_dataset))   # 20%
test_size = len(full_dataset) - train_size - val_size  # Remaining 

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

test_dataset.dataset.transform = t_test
train_dataset.transform = t_train
val_dataset.transform = t_val

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)



# model  = smp.DeepLabV3Plus(
#     encoder_name="resnet50",  # Use ResNet50 as encoder
#     encoder_weights="imagenet",  # Use ImageNet pretrained weights for the encoder
#     in_channels=3,  # Number of input channels (e.g., 3 for RGB images)
#     classes=2,  # Number of classes (including background)
# )
# model.segmentation_head[-1] = nn.Conv2d(model.segmentation_head[-1].in_channels, 2, kernel_size=1)
#model= build_unet()
model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=1, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])


loss_function = nn.BCEWithLogitsLoss()

max_lr = 1e-3
weight_decay = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                            steps_per_epoch=len(train_loader))

print("device is",device)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train_model(train_dataloader, val_dataloader, model, loss_fn, optimizer,scheduler):
    model.to(device)
    model.train()
    train_loss = 0.0
    train_iou = 0.0
    for batch, (images, masks) in enumerate(tqdm(train_dataloader)):
        images, masks = images.to(device), masks.to(device)
        #masks=masks.squeeze(1).long()
        #print("input image",images.shape)
     
        masks = masks.unsqueeze(1)
        output = model(images)
        #print("images output",output)
        threshold = 0.5
        #binary_predictions = (output > threshold).float()
        loss = loss_function(output, masks)
        #print("loss is",loss)
        loss.backward()
        optimizer.step() #update weight          
        optimizer.zero_grad() #reset gradient

        lrs.append(get_lr(optimizer))
        scheduler.step() 
        
        train_loss += loss.item()
        train_iou+= Iou_score(output,masks).item()

    train_loss /= len(train_loader)
    train_iou /= len(train_loader)
    train_loss_arrray.append(train_loss)
    train_iou_array.append(train_iou)

    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for images, masks in val_dataloader:
            images, masks = images.to(device), masks.to(device)
            #masks = masks.squeeze(1).long()
            masks = masks.unsqueeze(1)
            output = model(images)
            loss = loss_function(output, masks)

            val_loss += loss.item()
            val_iou += Iou_score(output, masks).item()

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_loss_array.append(val_loss)
    val_iou_array.append(val_iou)
    
    return train_loss, train_iou, val_loss, val_iou


history = {'train_loss' : train_loss_arrray, 'val_loss': val_loss_array,
               'train_miou' :train_iou_array, 'val_miou':val_iou_array,
            }

# def mIoU(pred_mask, mask, smooth=1e-10, n_classes=2):
#     with torch.no_grad():
#         pred_mask = F.softmax(pred_mask, dim=1)
#         pred_mask = torch.argmax(pred_mask, dim=1)
#         pred_mask = pred_mask.contiguous().view(-1)
#         mask = mask.contiguous().view(-1)

#         iou_per_class = []
#         for clas in range(0, n_classes): #loop per pixel class
#             true_class = pred_mask == clas
#             true_label = mask == clas

#             if true_label.long().sum().item() == 0: #no exist label in this loop
#                 iou_per_class.append(np.nan)
#             else:
#                 intersect = torch.logical_and(true_class, true_label).sum().float().item()
#                 union = torch.logical_or(true_class, true_label).sum().float().item()

#                 iou = (intersect + smooth) / (union +smooth)
#                 iou_per_class.append(iou)
#         return np.nanmean(iou_per_class)
    

def train():
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_iou, val_loss, val_iou = train_model(train_loader, val_loader, model, loss_function, optimizer, sched)
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
   
    # Create folder if not exist    
    os.makedirs("model", exist_ok=True)

    model_path = "model/unetc_model.pth"
    # Save model
    plot_loss(history)
    plot_score(history)
    torch.save(model, model_path)


if __name__ == "__main__":
    train()