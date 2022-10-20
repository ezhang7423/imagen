import glob
from pathlib import Path
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils

from PIL import Image

# helpers functions

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def normalize(threshold):

    def fn(image):
        image = torch.from_numpy(image)
        image[image != 0] = threshold
        image[image == 0] = -threshold
        
        return image.float()[None]

    return fn

# dataset and dataloader

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        threshold=1,
    ):
        super().__init__()
        self.folder = folder
        self.paths = glob.glob(f'{folder}/*.pt')


        self.transform = T.Compose([
            T.Lambda(normalize(threshold)),
            # T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Pad(1),
            # T.CenterCrop(image_size),
        ])

    def __len__(self):
        return len(self.paths)

    def preprocess(self, sample):
        colors = {
             1: 'blue',  # blue
             2: 'green',  # green
             3: 'red',  # red
             4: 'orange',  # orange
             5: 'purple',  # purple
             6: 'yellow',  # yellow
        }

        if sample['grid'].sum() != 0:
            ctx = 'There are blocks on'
            for z, y, x in zip(*sample['grid'].nonzero()):
                ctx += f' {z},{y},{x},{colors[sample["grid"][z,y,x]]}'
            
            ctx += '. Implement: '
        else:
            ctx = 'There are no blocks. Implement: '        

        return ctx + sample['dialog'].split('<Architect> ')[-1] #! add context
        
    def __getitem__(self, index):
        path = self.paths[index]
        # .split('<Architect> ')
        ret = torch.load(path)
        
        return self.transform(np.clip(ret['target_grid'] - ret['grid'], 0, 1)), self.preprocess(ret)

def get_images_dataloader(
    folder,
    *,
    batch_size,
    image_size,
    shuffle = True,
    cycle_dl = False,
    pin_memory = True
):
    ds = Dataset(folder, image_size)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory)

    if cycle_dl:
        dl = cycle(dl)
    return dl
