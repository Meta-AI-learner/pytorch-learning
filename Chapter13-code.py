import torch
from PIL import Image
import numpy as np
from torch.utils import data
import os
from torchvision import transforms
import PIL

transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                                ])

class DogCat(data.Dataset):
    def __init__(self, root, transforms = None):
        imgs = os.listdir(root) #图像的绝对路径
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transform = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        #array = np.array(pil_img)
        #data = torch.from_numpy(array)
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
'''
dataset = DogCat(r'./images/', transforms = transform)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), img.float().mean(), label)
'''
class new_DogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index) #注意return
        except PIL.UnidentifiedImageError:
            return None, None  

dataset = new_DogCat('imageswrong', transform)
from torch.utils.data.dataloader import default_collate #导入默认的拼接方式

def my_collate_fn(batch):
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    return default_collate(batch)

dataloader = data.DataLoader(dataset, 3, collate_fn=my_collate_fn)
for batch_data, batch_labels in dataloader:
    print(batch_data.size(), batch_labels.size())