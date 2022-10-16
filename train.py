import torch
from imagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from imagen_pytorch.data import Dataset
import torchvision
import os 
from torchtils.utils import datestr
import wandb

THRESHOLD = 0.01
wandb.init(project='imagen-training', entity='lang-diffusion', name=f'{THRESHOLD=}')

d = datestr()


os.makedirs(f'data/{d}',exist_ok=True)
os.chdir(f'data/{d}')

""""""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_grid(voxel, text=None):
    idx2color = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange', 5: 'purple', 6: 'yellow', -1: 'black'}
    vox = voxel.transpose(1, 2, 0)
    colors = np.empty(vox.shape, dtype=object)
    for i in range(vox.shape[0]):
        for j in range(vox.shape[1]):
            for k in range(vox.shape[2]):
                if vox[i, j, k] != 0:
                    colors[i][j][k] = str(idx2color[vox[i, j, k]])

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(projection='3d', )
    ax.voxels(vox, facecolors=colors, edgecolor='k', )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=11))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    ax.set_xticks(np.arange(0, 12, 1), minor=True)
    ax.set_yticks(np.arange(0, 12, 1), minor=True)
    ax.set_zticks(np.arange(0, 9, 1), minor=True)

    box = ax.get_position()
    box.x0 = box.x0 - 0.05
    box.x1 = box.x1 - 0.05
    box.y1 = box.y1 + 0.16
    box.y0 = box.y0 + 0.16
    ax.set_position(box)

    if text is not None:
        plt.annotate(text, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points',
                     verticalalignment='top', wrap=True)
    return fig
""""""


unet1 = Unet3D(dim = 128, dim_mults = (1, 2, 4)).cuda()


# elucidated imagen, which contains the unets above (base unet and super resoluting ones)

imagen = ElucidatedImagen(
    unets = unet1,
    image_sizes = 12,
    random_crop_sizes = None,
    num_sample_steps = 50,
    cond_drop_prob = 0.1,
    channels=1,
    sigma_min = 0.002,                          # min noise level
    sigma_max = 80,                      # max noise level, double the max noise level for upsampler
    sigma_data = 1,                           # standard deviation of data distribution
    rho = 7,                                    # controls the sampling schedule
    P_mean = -1.2,                              # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                                # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                               # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
    auto_normalize_img=False,
).cuda()

# mock videos (get a lot of this) and text encodings from large T5

# feed images into imagen, training each unet in the cascade
# for this example, only training unet 1

trainer = ImagenTrainer(imagen, checkpoint_path='./diffuser', checkpoint_every=10_000, split_valid_from_train=True).cuda()

dataset = Dataset('/data2/eddie/iglu/both', threshold=THRESHOLD)

trainer.add_train_dataset(dataset, batch_size = 18)
# working training loop

for i in range(200_000):
    loss = trainer.train_step(unet_number = 1)
    # print(f'loss: {loss}')
    wandb.log({'train/loss': loss})

    if not (i % 500):
        valid_loss = trainer.valid_step(unet_number = 1)
        wandb.log({'valid/loss': valid_loss})
        print(f'valid loss: {valid_loss}')

    if not (i % 1_000) and trainer.is_main: # is_main makes sure this can run in distributed
        images = trainer.sample(batch_size = 1, return_pil_images = True, texts=['Facing north, build three stacks of two orange blocks, then destroy the bottom orange block on all three stacks'], video_frames=9) # returns List[Image]
        images = torchvision.transforms.functional.center_crop(images, 11)
        images[images > 0] = 1
        images[images < 0] = 0        
        plot_grid(images[0, 0].cpu().numpy()).savefig(f'sample_{i}.png')



# trainer(videos, texts = texts)
# trainer.update()

# videos = trainer.sample(texts = texts, video_frames = 20) # extrapolating to 20 frames from training on 10 frames

# print(videos.shape) # (4, 3, 20, 32, 32)
