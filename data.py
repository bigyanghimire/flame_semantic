from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms 
import os
import cv2
from PIL import Image
import torch

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

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
        mask = torch.from_numpy(mask).float()
        return img, mask