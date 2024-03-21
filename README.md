# 3D Image Diffusion

This repo is a fork of Phil Wang's [Video Diffusion - Pytorch](https://github.com/lucidrains/video-diffusion-pytorch).

The original code was modified for 3D Image generation.

## Important Notes

- Default number of channels is 1.
- Time conditioning is off.
- Text conditioning is probably broken.
- `num_frames` now is a depth dimension: `B, C, D, H, W`. Since model assumes that `H=W` I think `D` has to be `D=H=W`, 
but this has to be tested.

## Install

```bash
$ git clone https://github.com/trsvchn/diff3d.git
$ cd diff3d
$ pip install .
```

## Usage

### Generate Image

```python
import torch
from diff3d import Unet3D, GaussianDiffusion

model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    num_frames=32,  # The len of depth dimension (ideally same as image_size)
    timesteps=1000,  # number of steps
    loss_type="l1",  # L1 or L2
)

image = torch.randn(1, 1, 32, 32, 32) # 3d image (batch, channels, depth, height, width) - normalized from -1 to +1
loss = diffusion(image)
loss.backward()
# after a lot of training

sampled_image = diffusion.sample(batch_size=4)
sampled_image.shape # (4, 1, 32, 32, 32)
```

### Image Inpainting

```python
import torch
from diff3d import Unet3D, GaussianDiffusion

model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    num_frames=32,  # The len of depth dimension (ideally same as image_size)
    timesteps=1000,  # number of steps
    loss_type="l1",  # L1 or L2
)

# New `inpaint` methods takes a batch of images and a batch of corresponding masks. Returns reconstructions os the same shape as images and masks. 
# By deafault writes intermediate inpaintings to `./inpainting` directory every 100 timesteps. Set None or False to disable.
images = torch.randn(4, 1, 32, 32, 32)  # (batch, channels, depth, height, width), normalized from 0 to 1 (e.g. after pil_to_tensor transform).
masks = ...  # binary masks (batch, channels, depth, height, width), where 1 - indicates inpainting regions.

inpainting = diffusion.inpaint(
    image=images,
    mask=masks,
    save_every=100,
)
inpainting.shape  # (4, 1, 32, 32, 32)
```

## Training

This repository also contains a handy `Trainer` class for training on a folder of `gifs`. Each `gif` must be of the correct dimensions `image_size` and `num_frames`.

```python
import torch
from diff3d import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    num_frames=64,  # The len of depth dimension (ideally same as image_size)
    timesteps=1000,  # number of steps
    loss_type="l1"  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    "./data",  # this folder path needs to contain all your training data, as .gif files, of correct image size
    train_batch_size=32,
    train_lr=1e-4,
    save_and_sample_every=1000,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
)

trainer.train()
```

Sample images (as `gif` files) will be saved to `./results` periodically, as are the diffusion model parameters.

## Citations

```bibtex
@software{,
  title = {Video Diffusion - Pytorch},
  author = {Phil Wang},
  year = {2023}
  license = {MIT},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lucidrains/video-diffusion-pytorch}},
  commit = {f68f31eaa94c2b9987571136e6bb8c4f52960eef}
}
```

```bibtex
@misc{ho2022video,
  title   = {Video Diffusion Models}, 
  author  = {Jonathan Ho and Tim Salimans and Alexey Gritsenko and William Chan and Mohammad Norouzi and David J. Fleet},
  year    = {2022},
  eprint  = {2204.03458},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

```bibtex
@misc{Saharia2022,
    title   = {Imagen: unprecedented photorealism × deep level of language understanding},
    author  = {Chitwan Saharia*, William Chan*, Saurabh Saxena†, Lala Li†, Jay Whang†, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho†, David Fleet†, Mohammad Norouzi*},
    year    = {2022}
}
```
