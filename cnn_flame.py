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
random_seed = 89
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)



batch_size=3
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png')) 

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
 
        if self.transform:
            aug = self.transform(image=image,mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        return img, mask
        
    
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((256, 256)),
                                transforms.Normalize(mean, std)])
u_transform=A.Compose([A.Resize(256, 256)])
full_dataset = CustomDataset('dataset/Images', 'dataset/Masks',u_transform)

t_train =A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()])
t_val = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])
t_test = A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)

train_size = int(0.6 * len(full_dataset))  # 60% 
val_size = int(0.2 * len(full_dataset))   # 20%
test_size = len(full_dataset) - train_size - val_size  # Remaining 

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

test_dataset.dataset.transform = t_test
train_dataset.transform = t_train
val_dataset.transform = t_val

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("len of dataset",len(train_loader.dataset))
print("len of val dataset",len(val_loader.dataset))


model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=2, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

criterion = nn.CrossEntropyLoss()
max_lr = 1e-3
weight_decay = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is",device)
lrs = []
epochs=1
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                            steps_per_epoch=len(train_loader))

test_iou = [];

train_iou_array = [];
train_loss_arrray = [];

test_iou_array = [];
test_loss_arrray = [];

val_iou_array = [];
val_loss_array = [];
#running_loss=0

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
        masks=masks.squeeze(1).long()
        output = model(images)
        loss = criterion(output, masks)
        #iou_score = mIoU(output, masks)

        #print("iou score compute_iou",iou_score)

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
            masks = masks.squeeze(1).long()

            output = model(images)
            loss = criterion(output, masks)

            val_loss += loss.item()
            val_iou += Iou_score(output, masks).item()

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_loss_array.append(val_loss)
    val_iou_array.append(val_iou)
    
    return train_loss, train_iou, val_loss, val_iou

        # if batch % 500 == 0:
        #     # train_loss = loss.item()
        #     # test_loss  = test_accuracy(test_loader,model,loss_fn)
        #     # print("test loss is",test_loss)
        #     # print(f"Train_loss: {train_loss:>7f}  Test_loss:{test_loss:>7f} Test_acc:{testing_accuracy}%")

# def test_accuracy(dataloader, model, loss_fn):
#     test_loss = 0.0
#     iou_score = 0

#     with torch.no_grad():
#         for batch, (images, masks) in enumerate(dataloader):
#             images, masks = images.to(device), masks.to(device)
#             masks=masks.squeeze(1).long()
#             outputs = model(images)
#             iou_score += mIoU(outputs, masks)
#             test_iou.append(iou_score/len(dataloader))

#             loss = loss_fn(outputs, masks)
#             test_loss += loss.item()

#         #test_iou.append(iou_score/len(dataloader))

#         test_loss /= len(dataloader.dataset)
#         test_loss_arrray.append(test_loss)
#         return iou_score/len(dataloader)

def Iou_score(outputs,masks):
    iou_metric = JaccardIndex(task='multiclass', num_classes=2).to(device)
    pred_masks = torch.argmax(outputs, dim=1)
    iou_metric(pred_masks, masks)
    return iou_metric.compute()

def test2():
    model = torch.load('model/model_final2.pth',map_location=torch.device(device))
    model.eval()
    iou_metric = JaccardIndex(task='multiclass', num_classes=2).to(device)
    total_time = 0
    test_loss=0
    with torch.no_grad():
        for images, masks in test_loader:
            start_time = time.time()  # Start time for measuring fps
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1).long()
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            pred_masks = torch.argmax(outputs, dim=1)
            iou_metric(pred_masks, masks)
            end_time = time.time()  # End time for measuring fps
            total_time += end_time - start_time
    
    
    test_loss /= len(test_loader.dataset)
    test_loss_arrray.append(test_loss)
    # Calculate frames per second (fps)
    total_images = len(test_loader.dataset)
    avg_time_per_image = total_time / total_images
    fps = 1 / avg_time_per_image
    
    miou = iou_metric.compute()
    test_iou_array.append(miou.item())
    print("Test IOU is: ",miou.item())
    print("FPS is: ",fps)
    print("Test Loss is: ",test_loss)
    return miou, fps

history = {'train_loss' : train_loss_arrray, 'val_loss': test_loss_arrray,
               'train_miou' :train_iou_array, 'val_miou':test_iou_array,
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
    
def test():
    start_time = time.time()
    model = torch.load('model/model_final2.pth',map_location=torch.device('cpu'))
    model.eval()
    image, mask = Image.open('oo.jpg') , Image.open('dataset/Images/image_0.jpg') 
    image=transform(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():
        start_time = time.time()
        image = image.unsqueeze(0)
        output = model(image)
        torch.set_printoptions(profile="full")
        #print("outout is",output)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
        end_time = time.time()  # End time for measuring fps
        total_time = end_time - start_time
        print("total time is",total_time)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
        ax1.imshow(masked)
        plt.show()  

def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot( history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU',  marker='*')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def train():
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_iou, val_loss, val_iou = train_model(train_loader, val_loader, model, criterion, optimizer, sched)
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # #train_model(train_loader, model, criterion, optimizer,sched)
        # #test_accuracy_score=test_accuracy(test_loader,model,criterion)
        # iou2,fps= test_accuracy_2(test_loader,model)
        # #print("test accuracy",test_accuracy_score)
        # print("test accuracy2",iou2)
        # print("fps is",fps)
        # print("train loss aray is",train_loss_arrray)
        # print("test loss aray is",test_loss_arrray)
        # print("test iou aray is",test_iou_array)
        # print("train iou aray is",train_iou_array)

   
    # Create folder if not exist    
    os.makedirs("model", exist_ok=True)

    model_path = "model/model_final2.pth"
    # Save model
    plot_loss(history)
    plot_score(history)
    torch.save(model, model_path)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        train()
    elif mode == "test":
       test2()
    else:
        print("Please use train or test")
        sys.exit(1)