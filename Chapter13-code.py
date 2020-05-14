import torch
from PIL import Image
import numpy as np
from torch.utils import data
import os

class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root) #图像的绝对路径
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img = Image.open(img_path)
        array = np.array(pil_img)
        data = torch.from_numpy(array)
        return data, label

    def __len__(self):
        return len(self.imgs)

dataset = DogCat(r'./images/')
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), img.float().mean(), label)