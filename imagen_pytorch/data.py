import glob
from pathlib import Path
from functools import partial

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

def normalize(image):

    image = torch.from_numpy(image)
    image[image != 0] = 1
    image[image == 0] = -1
    
    return image.float()[None]

# dataset and dataloader

class Dataset(Dataset):
    def __init__(
        self,
        folder,
    ):
        super().__init__()
        self.folder = folder
        self.paths = glob.glob(f'{folder}/*.pt')


        self.transform = T.Compose([
            T.Lambda(normalize),
            # T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Pad(1),
            # T.CenterCrop(image_size),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        # .split('<Architect> ')
        ret = torch.load(path)
        
        return self.transform(ret['target_grid']), ret['dialog'].split('<Architect> ')[-1]

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
