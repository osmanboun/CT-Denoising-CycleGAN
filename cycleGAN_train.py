
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from multiprocessing import cpu_count

from os import listdir, makedirs
from os.path import isdir, join 
from torch.utils.data import Dataset, DataLoader
from torch.nn import init

from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.cuda.amp import autocast, GradScaler


# Functions for calculating PSNR, SSIM
# Peak Signal-to-Noise Ratio
def psnr(A, ref):
    """
    Compute PSNR between two 2D numpy arrays A and ref.
    Both arrays are clipped at -1000 (HU floor). Images are normalized to [0,1]
    using the min/max of the reference image and PSNR is computed with data_range=1.0.
    Returns a float (higher is better). Handles constant images.
    """
    A = A.copy()
    ref = ref.copy()
    A[A < -1000] = -1000
    ref[ref < -1000] = -1000

    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))
    denom = ref_max - ref_min
    if denom <= 0:
        return 100.0

    ref_n = (ref - ref_min) / denom
    A_n = (A - ref_min) / denom

    return peak_signal_noise_ratio(ref_n, A_n, data_range=1.0)

# Structural similarity index
def ssim(A, ref):
    """
    Compute SSIM between two 2D numpy arrays A and ref.
    Both arrays are clipped at -1000 and normalized to [0,1] using the
    reference image range. Returns a float in [-1,1] (higher is better).
    """
    A = A.copy()
    ref = ref.copy()
    A[A < -1000] = -1000
    ref[ref < -1000] = -1000

    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))
    denom = ref_max - ref_min
    if denom <= 0:
        return 1.0

    ref_n = (ref - ref_min) / denom
    A_n = (A - ref_min) / denom

    # structural_similarity expects 2D arrays and a proper data_range
    return structural_similarity(ref_n, A_n, data_range=1.0)


# Torch-based SSIM (differentiable) and gradient loss for training
import math

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_torch(img1, img2, window_size=11, val_range=1.0):
    """
    Compute mean SSIM over batch using a Gaussian window. Returns a scalar in [-1,1].
    Assumes img1 and img2 are torch tensors shaped (B, C, H, W) and scaled to [0, val_range].
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device).type(img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def gradient_loss(pred, target):
    """
    L1 loss between image gradients (Sobel) of pred and target. Works on BxCxHxW tensors.
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)

    channels = pred.shape[1]
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    grad_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
    grad_target_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
    grad_target_y = F.conv2d(target, sobel_y, padding=1, groups=channels)

    loss = (grad_pred_x - grad_target_x).abs().mean() + (grad_pred_y - grad_target_y).abs().mean()
    return loss


# Initialize parameters of neural networks
def init_weights(net):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        
    print('Initialize network.')
    net.apply(init_func)
    
    
# Set 'requires_grad' of the networks
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# Calculate average loss during one epoch
class Mean:
    def __init__(self):
        self.numel = 0
        self.mean = 0
    
    def __call__(self, val):
        self.mean = self.mean * (self.numel / (self.numel + 1)) + val / (self.numel + 1)
        self.numel += 1
    
    def result(self):
        return self.mean


# Image pool for storing previously generated images (used to update discriminator)
class ImagePool:
    """Initialize an image buffer that stores previously generated images.

    This buffer enables the discriminator to be trained on a history of generated images
    rather than only the most recent ones, which stabilizes GAN training (CycleGAN trick).
    """
    def __init__(self, pool_size=50):
        self.pool_size = int(pool_size)
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        Return images for discriminator training. Accepts a batch tensor (B,C,H,W).
        For each image in the batch: if the pool isn't full, store and return image.
        Otherwise, with probability 0.5 return a previously stored image (and replace it with the new one),
        or return the new image directly.
        """
        if self.pool_size == 0:
            return images

        return_images = []
        for img in images:
            img = img.unsqueeze(0)
            if self.num_imgs < self.pool_size:
                # store and return
                self.images.append(img.clone().detach())
                self.num_imgs += 1
                return_images.append(img)
            else:
                if random.random() > 0.5:
                    # use previously stored image
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone().detach()
                    # replace the stored image with the new one
                    self.images[idx] = img.clone().detach()
                    return_images.append(tmp)
                else:
                    return_images.append(img)
        return torch.cat(return_images, dim=0)


# CT dataset
class CT_Dataset(Dataset):
    def __init__(self, path, transform, shuffle=True):
        # Path of 'full_dose' and 'quarter_dose' folders
        self.path_full = join(path, 'full_dose')
        self.path_quarter = join(path, 'quarter_dose')
        self.transform = transform

        # File list of full dose data
        self.file_full = list()
        for file_name in sorted(listdir(self.path_full)):
            self.file_full.append(file_name)
            
        if shuffle:
            random.seed(0)
            random.shuffle(self.file_full)
        
        # File list of quarter dose data
        self.file_quarter = list()
        for file_name in sorted(listdir(self.path_quarter)):
            self.file_quarter.append(file_name)
    
    def __len__(self):
        return min(len(self.file_full), len(self.file_quarter))
    
    def __getitem__(self, idx):
        # Load full dose/quarter dose data
        x_F = np.load(join(self.path_full, self.file_full[idx]))
        x_Q = np.load(join(self.path_quarter, self.file_quarter[idx]))

        # Convert to HU scale
        x_F = (x_F - 0.0192) / 0.0192 * 1000
        x_Q = (x_Q - 0.0192) / 0.0192 * 1000

        # Normalize images
        x_F[x_F < -1000] = -1000
        x_Q[x_Q < -1000] = -1000

        x_F = x_F / 4000
        x_Q = x_Q / 4000

        # Apply transform
        x_F = self.transform(x_F)
        x_Q = self.transform(x_Q)

        file_name = self.file_quarter[idx]

        return x_F, x_Q, file_name
  

# Transform for the random crop
class RandomCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        """
        Robust random crop that handles inputs as numpy arrays or torch tensors,
        supports channel-last (H, W, C) and channel-first (C, H, W) formats,
        and pads images smaller than the patch size using reflection padding.
        Returns a tensor with shape (C, patch_size, patch_size).
        """
        # If input is a numpy array, convert to torch tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

        # Ensure tensor type
        if not torch.is_tensor(img):
            img = torch.tensor(img, dtype=torch.float32)

        # Handle 2D images (H x W) -> add channel dim
        if img.dim() == 2:
            img = img.unsqueeze(0)

        # Handle channel-last (H x W x C) -> convert to C x H x W
        elif img.dim() == 3 and img.shape[-1] <= 4 and img.shape[0] > 4:
            img = img.permute(2, 0, 1)

        # Now img is expected to be C x H x W
        C, H, W = img.shape

        # Pad if image smaller than patch size (reflect padding)
        pad_h = max(0, self.patch_size - H)
        pad_w = max(0, self.patch_size - W)
        if pad_h > 0 or pad_w > 0:
            # F.pad expects (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            C, H, W = img.shape

        # Random crop coordinates (safe since we padded if necessary)
        i = random.randint(0, H - self.patch_size)
        j = random.randint(0, W - self.patch_size)

        return img[:, i:i + self.patch_size, j:j + self.patch_size]



# Make dataloader for training/test
def make_dataloader(path, train_batch_size=1, is_train=True, num_workers=None):
    """
    Create a DataLoader for training or testing.

    Small I/O improvements:
    - Automatically choose a sensible number of workers (defaults to cpu_count()-1 up to 4).
    - Use pin_memory only when CUDA is available.
    - Enable persistent_workers when num_workers>0 to avoid worker spawn overhead each epoch.
    - Set a small prefetch_factor when using multiple workers.
    """
    # Path of 'train' and 'test' folders
    dataset_path = join(path, 'train') if is_train else join(path, 'test')

    # Choose number of workers if not explicitly provided
    if num_workers is None:
        try:
            n_workers = max(0, min(4, cpu_count() - 1))
        except Exception:
            n_workers = 2
    else:
        n_workers = max(0, int(num_workers))

    pin_memory = torch.cuda.is_available()
    persistent = True if n_workers > 0 else False

    # Transform for training data: convert to tensor, random horizontal/verical flip, random crop
    if is_train:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            RandomCrop(128)
        ])
        train_dataset = CT_Dataset(dataset_path, train_transform)
        dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if n_workers > 0 else 2
        )
    else:
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        test_dataset = CT_Dataset(dataset_path, test_transform)
        dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent
        )

    return dataloader

  

class DenseLayer(nn.Module):
    """
    Single dense layer used inside a DenseBlock. Takes input features and produces growth_channels features,
    then concatenates them to the input (DenseNet-style).
    """
    def __init__(self, in_channels, growth_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_channels, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """
    Dense block composed of several DenseLayer layers. After growth layers, a 1x1 conv (local feature fusion)
    reduces the concatenated channels back to the input channels so the block has a residual-friendly shape.
    """
    def __init__(self, in_channels, growth_channels=32, num_layers=5):
        super(DenseBlock, self).__init__()
        self.num_layers = int(num_layers)
        self.growth = int(growth_channels)
        self.layers = nn.ModuleList()
        ch = in_channels
        for i in range(self.num_layers):
            self.layers.append(DenseLayer(ch, self.growth))
            ch += self.growth
        # local feature fusion: reduce ch -> in_channels
        self.lff = nn.Conv2d(ch, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.lff(out)
        return out


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB) as used in ESRGAN / RRDBNet.
    It stacks `num_blocks` DenseBlocks and applies a residual-scaling to the dense output
    before adding back to the input: out = x + res_scale * F(x)
    Defaults reduced for lower compute: growth_channels=16, num_layers=3, num_blocks=2.
    """
    def __init__(self, channels, growth_channels=16, num_layers=3, num_blocks=2, res_scale=0.2):
        super(RRDB, self).__init__()
        self.blocks = nn.ModuleList([DenseBlock(channels, growth_channels, num_layers) for _ in range(num_blocks)])
        self.res_scale = float(res_scale)

    def forward(self, x):
        out = x
        for b in self.blocks:
            out = b(out)
        return x + out * self.res_scale

class Upsample(nn.Module):
    '''
    Upsample module that performs bilinear upsampling followed by convolution, batch normalization, and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels. If not provided, it will be set to the same as in_channels.

    Attributes:
        up (torch.nn.Upsample): Bilinear upsampling layer.
        conv (torch.nn.Conv2d): Convolutional layer.
        norm (torch.nn.BatchNorm2d): Batch normalization layer.
    '''

    def __init__(self, in_channels, out_channels=None):
        super(Upsample, self).__init__()
        if not out_channels:
            # If out_channels is not provided, set it to the same as in_channels
            out_channels = in_channels
        # Bilinear upsampling layer
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Convolutional layer
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Batch normalization layer
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        '''
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        # Leaky ReLU activation function
        x = torch.nn.LeakyReLU(0.1)(x)
        return x


class Downsample(nn.Module):
    '''
    A class representing a downsampling module.

    This module performs downsampling on the input tensor using a convolutional layer with a stride of 2.

    Args:
        in_channels (int): The number of input channels.

    Attributes:
        in_channels (int): The number of input channels.
        conv (torch.nn.Conv2d): The convolutional layer used for downsampling.
    '''

    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.in_channels = in_channels
        self.conv = torch.nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        '''
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying convolution.

        '''
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)

        return x

class Generator(nn.Module):
    """
    Simplified encoder-decoder Generator using RRDB blocks and single skip connection per scale.

    Structure:
      - conv_in -> down_blocks (RRDBs + optional Downsample)
      - mid_block (RRDB)
      - up_blocks: for each scale, (optional Upsample) -> concat with corresponding encoder skip -> 1x1 reduce -> RRDBs
      - conv_out + global residual
    """
    def __init__(self, in_channels, out_channels, ngf, ch_mult=(1, 2, 4, 8), num_res_blocks=3, rrdb_gc=16, rrdb_num_layers=3, rrdb_num_blocks=2):
        super(Generator, self).__init__()
        assert in_channels == out_channels, 'The number of input channels should be equal to the number of output channels.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # RRDB configuration (reduced for compute)
        self.rrdb_gc = rrdb_gc
        self.rrdb_num_layers = rrdb_num_layers
        self.rrdb_num_blocks = rrdb_num_blocks

        self.down_blocks = nn.ModuleList()

        # initial conv
        self.conv_in = nn.Conv2d(in_channels, ngf, kernel_size=3, stride=1, padding=1)

        # Build down blocks: for each level, optionally expand channels with 1x1, apply several RRDBs,
        # and optionally downsample at the end.
        in_ch_mult = (1,) + tuple(ch_mult)
        for level in range(self.num_resolutions):
            layers = nn.ModuleList()
            block_in_channels = ngf * in_ch_mult[level]
            block_out_channels = ngf * ch_mult[level]

            # channel expansion if needed
            if block_in_channels != block_out_channels:
                layers.append(nn.Conv2d(block_in_channels, block_out_channels, kernel_size=1, stride=1, padding=0))
                block_in_channels = block_out_channels

            # RRDBs (use reduced complexity params)
            for _ in range(self.num_res_blocks):
                layers.append(RRDB(block_in_channels, growth_channels=min(self.rrdb_gc, max(8, block_in_channels // 4)), num_layers=self.rrdb_num_layers, num_blocks=self.rrdb_num_blocks))

            # downsample except for last level
            if level != self.num_resolutions - 1:
                layers.append(Downsample(block_out_channels))

            self.down_blocks.append(layers)

        # bottleneck (reduced RRDB)
        self.mid_block = RRDB(ngf * ch_mult[-1], growth_channels=min(self.rrdb_gc, max(8, (ngf * ch_mult[-1]) // 4)), num_layers=self.rrdb_num_layers, num_blocks=self.rrdb_num_blocks)

        # Build up blocks: at each level we will (optionally) upsample, concatenate with corresponding skip, reduce channels,
        # and apply RRDBs. We'll store them so forward pass can run deterministically.
        self.up_blocks = nn.ModuleList()
        prev_channels = ngf * ch_mult[-1]
        # encoder channel sizes (skip features) in order of append during down pass
        encoder_channels = [ngf] + [ngf * m for m in ch_mult]

        for level in reversed(range(self.num_resolutions)):
            layers = nn.ModuleList()
            # If not the deepest level, we need to upsample previous output first
            if level != self.num_resolutions - 1:
                layers.append(Upsample(prev_channels, out_channels=prev_channels))

            skip_channels = encoder_channels[level + 1]
            concat_ch = prev_channels + skip_channels
            out_ch = ngf * ch_mult[level]
            # conv to reduce concatenated channels back to the desired out_ch
            layers.append(nn.Conv2d(concat_ch, out_ch, kernel_size=1, stride=1, padding=0))
            # followed by several RRDBs operating at out_ch
            for _ in range(self.num_res_blocks):
                layers.append(RRDB(out_ch, growth_channels=min(self.rrdb_gc, max(8, out_ch // 4)), num_layers=self.rrdb_num_layers, num_blocks=self.rrdb_num_blocks))

            prev_channels = out_ch
            # prepend since we iterate reversed and want up_blocks[0] to correspond to the first up level
            self.up_blocks.insert(0, layers)

        self.conv_out = nn.Conv2d(ngf, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder: store skip connections BEFORE downsampling so they match the upsampled shapes later
        skips = []
        h = self.conv_in(x)
        for level in range(self.num_resolutions):
            layers = self.down_blocks[level]
            # if last layer is Downsample, apply all but last, store skip, then downsample for next level
            if len(layers) > 0 and isinstance(layers[-1], Downsample):
                for layer in layers[:-1]:
                    h = layer(h)
                skips.append(h)
                h = layers[-1](h)
            else:
                for layer in layers:
                    h = layer(h)
                skips.append(h)

        # Bottleneck
        h = self.mid_block(h)

        # Decoder / up path: for each level, optionally upsample, concat with corresponding skip, reduce and RRDBs
        for level in range(self.num_resolutions - 1, -1, -1):
            layers = self.up_blocks[level]
            idx = 0
            if isinstance(layers[0], Upsample):
                h = layers[0](h)
                idx = 1
            # pop corresponding skip (the last appended)
            skip = skips.pop()
            # concatenate and reduce
            h = torch.cat([h, skip], dim=1)
            h = layers[idx](h)
            # remaining layers are RRDBs
            for layer in layers[idx + 1:]:
                h = layer(h)

        h = self.conv_out(h)
        # global residual connection
        h = h + x
        return h

# Discriminator (PatchGAN)
# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    '''
    Discriminator network for CycleGAN.

    Args:
        in_channels (int): Number of input channels.
        ndf (int): Number of discriminator filters.

    '''

    def __init__(self, in_channels, ndf=32):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.ndf = ndf

        # Convolutional layers (same configuration as original)
        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1)
        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1)

        # Declare InstanceNorm layers once in __init__ to avoid recreating them each forward
        self.norm2 = nn.InstanceNorm2d(ndf * 2, affine=False)
        self.norm3 = nn.InstanceNorm2d(ndf * 4, affine=False)
        self.norm4 = nn.InstanceNorm2d(ndf * 8, affine=False)

    def forward(self, x, threshold=0.2):
        '''
        Forward pass of the discriminator network.

        Args:
            x (torch.Tensor): Input tensor.
            threshold (float): Leaky ReLU threshold.

        Returns:
            torch.Tensor: Output tensor.

        '''
        h = self.conv1(x)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv2(h)
        h = self.norm2(h)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv3(h)
        h = self.norm3(h)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv4(h)
        h = self.norm4(h)
        h = nn.functional.leaky_relu(h, threshold)

        h = self.conv5(h)
        return h


# Training function
def train(
    path_checkpoint='./CT_denoising',
    model_name='cyclegan_v1',
    path_data='../data/AAPM_data',
    batch_size=16,
    lambda_cycle=10,
    lambda_iden=5,
    beta1=0.5,
    beta2=0.999,
    num_epoch=100,
    g_channels=32,
    d_channels=64,
    ch_mult=[1, 2, 4, 8],
    num_res_blocks=3,
    lr=2e-4,
    use_checkpoint=False
):
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path for saving the checkpoint
    if not isdir(path_checkpoint):
        makedirs(path_checkpoint)

    # Path for saving results
    path_result = join(path_checkpoint, model_name)
    if not isdir(path_result):
        makedirs(path_result)

    # Make dataloaders
    train_dataloader = make_dataloader(path_data, batch_size)

    # Make generators (G_F2Q: full to quarter / G_Q2F: quarter to full)
    G_F2Q = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)
    G_Q2F = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)

    # Make discriminators (D_F: distinguish real/fake full dose images / D_Q: distinguish real/fake quarter dose images)
    D_F = Discriminator(1, d_channels).to(device)
    D_Q = Discriminator(1, d_channels).to(device)

    # Convert models to channels_last memory format for improved conv throughput on modern GPUs.
    # Wrapped in try/except for older PyTorch versions or unsupported devices.
    if torch.cuda.is_available():
        try:
            G_F2Q = G_F2Q.to(memory_format=torch.channels_last)
            G_Q2F = G_Q2F.to(memory_format=torch.channels_last)
            D_F = D_F.to(memory_format=torch.channels_last)
            D_Q = D_Q.to(memory_format=torch.channels_last)
        except Exception:
            # If conversion fails, continue with default memory format.
            pass


    # Image pools for discriminator history (CycleGAN paper suggests a small buffer, e.g. 50)
    pool_F = ImagePool(pool_size=50)  # stores fake full images (x_QF)
    pool_Q = ImagePool(pool_size=50)  # stores fake quarter images (x_FQ)

    # Make optimizers
    G_optim = torch.optim.Adam(itertools.chain(G_F2Q.parameters(), G_Q2F.parameters()), lr, betas=(beta1, beta2))
    D_optim = torch.optim.Adam(itertools.chain(D_F.parameters(), D_Q.parameters()), lr, betas=(beta1, beta2))

    # Define loss functions
    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    iden_loss = nn.L1Loss()

    # Loss functions
    loss_name = ['G_adv_loss_F',
                'G_adv_loss_Q',
                'G_cycle_loss_F',
                'G_cycle_loss_Q',
                'G_iden_loss_F',
                'G_iden_loss_Q',
                'D_adv_loss_F',
                'D_adv_loss_Q']

    # SSIM and gradient loss weights (additive to generator loss)
    lambda_ssim = 1.0
    lambda_grad = 1.0


    # Mixed precision scalers
    scaler_G = GradScaler()
    scaler_D = GradScaler()

    if use_checkpoint:
        # If a checkpoint exists, load the state of the model and optimizer from the checkpoint
        checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'))
        G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
        G_F2Q.load_state_dict(checkpoint['G_F2Q_state_dict'])
        D_Q.load_state_dict(checkpoint['D_Q_state_dict'])
        D_F.load_state_dict(checkpoint['D_F_state_dict'])
        G_optim.load_state_dict(checkpoint['G_optim_state_dict'])
        D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
    else:
        # If no checkpoint exists, initialize the weights of the models
        init_weights(G_F2Q)
        init_weights(G_Q2F)
        init_weights(D_F)
        init_weights(D_Q)
        
    # Set the initial trained epoch as 0
    trained_epoch = 0

    # Validation tracking
    best_val_psnr = -1e9
    val_psnr_list = []
    val_ssim_list = []

    # Initialize a dictionary to store the losses
    losses_list = {name: list() for name in loss_name}
    print('Start from random initialized model')

    # Start the training loop
    for epoch in tqdm(range(trained_epoch, num_epoch), desc='Epoch', total=num_epoch, initial=trained_epoch):
        # Initialize a dictionary to store the mean losses for this epoch
        losses = {name: Mean() for name in loss_name}

        for x_F, x_Q, _ in tqdm(train_dataloader, desc='Step'):
            # Move the data to the device (GPU or CPU) and convert to channels_last memory format for faster conv operations.
            x_F = x_F.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            x_Q = x_Q.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

            # Set 'requires_grad' of the discriminators as 'False' to avoid computing gradients of the discriminators
            set_requires_grad([D_F, D_Q], False)
            # Update generators: perform forward and loss computations under autocast
            with autocast(enabled=torch.cuda.is_available()):
                # Generate fake images using the generators
                x_FQ = G_F2Q(x_F)
                x_QF = G_Q2F(x_Q)

                # Generate cyclic images using the generators
                x_QFQ = G_F2Q(x_QF)
                x_FQF = G_Q2F(x_FQ)

                # Generate identity images using the generators
                x_QQ = G_F2Q(x_Q)
                x_FF = G_Q2F(x_F)

                # Calculate adversarial losses (discriminator calls are inside autocast)
                G_adv_loss_F = adv_loss(D_F(x_QF), torch.ones_like(D_F(x_QF)))
                G_adv_loss_Q = adv_loss(D_Q(x_FQ), torch.ones_like(D_Q(x_FQ)))

                # Calculate cycle losses
                G_cycle_loss_F = cycle_loss(x_FQF, x_F)
                G_cycle_loss_Q = cycle_loss(x_QFQ, x_Q)

                # Calculate identity losses
                G_iden_loss_F = iden_loss(x_FF, x_F)
                G_iden_loss_Q = iden_loss(x_QQ, x_Q)

                # SSIM losses (1 - ssim so lower is better)
                # ssim_torch expects images scaled to [0,1] (we assume inputs are already normalized)
                G_ssim_F = 1.0 - ssim_torch(x_QF, x_F)
                G_ssim_Q = 1.0 - ssim_torch(x_FQ, x_Q)

                # Gradient losses (edge-aware)
                G_grad_F = gradient_loss(x_QF, x_F)
                G_grad_Q = gradient_loss(x_FQ, x_Q)

                # Calculate total losses
                G_adv_loss = G_adv_loss_F + G_adv_loss_Q
                G_cycle_loss = G_cycle_loss_F + G_cycle_loss_Q
                G_iden_loss = G_iden_loss_F + G_iden_loss_Q
                G_ssim_loss = G_ssim_F + G_ssim_Q
                G_grad_loss = G_grad_F + G_grad_Q

                G_total_loss = G_adv_loss + lambda_cycle * (G_cycle_loss) + lambda_iden * (G_iden_loss) + lambda_ssim * (G_ssim_loss) + lambda_grad * (G_grad_loss)

            # Update the generators (with GradScaler if available)
            G_optim.zero_grad()
            if scaler_G is not None:
                scaler_G.scale(G_total_loss).backward()
                scaler_G.step(G_optim)
                scaler_G.update()
            else:
                G_total_loss.backward()
                G_optim.step()
            
            # Set 'requires_grad' of the discriminators as 'True'
            set_requires_grad([D_F, D_Q], True)
            # Calculate adversarial losses for the discriminators
            # Use image pools (history) for generated images to stabilize discriminator training
            fake_QF_for_D = pool_F.query(x_QF.detach())
            fake_FQ_for_D = pool_Q.query(x_FQ.detach())

            with autocast(enabled=torch.cuda.is_available()):
                real_F_out = D_F(x_F)
                fake_F_out = D_F(fake_QF_for_D)
                D_adv_loss_F = adv_loss(real_F_out, torch.ones_like(real_F_out)) + adv_loss(fake_F_out, torch.zeros_like(fake_F_out))

                real_Q_out = D_Q(x_Q)
                fake_Q_out = D_Q(fake_FQ_for_D)
                D_adv_loss_Q = adv_loss(real_Q_out, torch.ones_like(real_Q_out)) + adv_loss(fake_Q_out, torch.zeros_like(fake_Q_out))

            D_total_loss_F = D_adv_loss_F / 2.0
            D_total_loss_Q = D_adv_loss_Q / 2.0

            # Update the discriminators (with GradScaler if available)
            D_optim.zero_grad()
            D_total_loss = D_total_loss_F + D_total_loss_Q
            if scaler_D is not None:
                scaler_D.scale(D_total_loss).backward()
                scaler_D.step(D_optim)
                scaler_D.update()
            else:
                D_total_loss.backward()
                D_optim.step()

            # Calculate the average loss during one epoch
            losses['G_adv_loss_F'](G_adv_loss_F.detach())
            losses['G_adv_loss_Q'](G_adv_loss_Q.detach())
            losses['G_cycle_loss_F'](G_cycle_loss_F.detach())
            losses['G_cycle_loss_Q'](G_cycle_loss_Q.detach())
            losses['G_iden_loss_F'](G_iden_loss_F.detach())
            losses['G_iden_loss_Q'](G_iden_loss_Q.detach())
            losses['D_adv_loss_F'](D_adv_loss_F.detach())
            losses['D_adv_loss_Q'](D_adv_loss_Q.detach())
    
        for name in loss_name:
            losses_list[name].append(losses[name].result())

        # Validation evaluation (compute PSNR/SSIM on test set)
        val_loader = make_dataloader(path_data, train_batch_size=1, is_train=False)
        G_F2Q.eval(); G_Q2F.eval()
        psnr_vals = []
        ssim_vals = []
        with torch.no_grad():
            for v_F, v_Q, _ in val_loader:
                v_F = v_F.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                v_Q = v_Q.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                pred = G_Q2F(v_Q)
                pred_np = pred.squeeze().cpu().numpy()
                ref_np = v_F.squeeze().cpu().numpy()
                psnr_vals.append(psnr(pred_np.copy(), ref_np.copy()))
                ssim_vals.append(ssim(pred_np.copy(), ref_np.copy()))
        avg_psnr = float(np.mean(psnr_vals)) if len(psnr_vals) > 0 else 0.0
        avg_ssim = float(np.mean(ssim_vals)) if len(ssim_vals) > 0 else 0.0
        val_psnr_list.append(avg_psnr)
        val_ssim_list.append(avg_ssim)

        # Save best model by validation PSNR
        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            torch.save({'epoch': epoch + 1, 'G_F2Q_state_dict': G_F2Q.state_dict(), 'G_Q2F_state_dict': G_Q2F.state_dict(),
                        'D_F_state_dict': D_F.state_dict(), 'D_Q_state_dict': D_Q.state_dict(),
                        'G_optim_state_dict': G_optim.state_dict(), 'D_optim_state_dict': D_optim.state_dict()},
                       join(path_checkpoint, model_name + '_best.pth'))
        G_F2Q.train(); G_Q2F.train()
        G_F2Q.train(); G_Q2F.train()

        # Save the trained model and list of losses
        torch.save({'epoch': epoch + 1, 'G_F2Q_state_dict': G_F2Q.state_dict(), 'G_Q2F_state_dict': G_Q2F.state_dict(),
                        'D_F_state_dict': D_F.state_dict(), 'D_Q_state_dict': D_Q.state_dict(),
                        'G_optim_state_dict': G_optim.state_dict(), 'D_optim_state_dict': D_optim.state_dict()}, join(path_checkpoint, model_name + '.pth'))
        for name in loss_name:
            torch.save(losses_list[name], join(path_result, name + '.npy'))
            
        # Save validation metrics if they exist
        try:
            np.save(join(path_result, 'val_psnr.npy'), np.array(val_psnr_list))
            np.save(join(path_result, 'val_ssim.npy'), np.array(val_ssim_list))
        except NameError:
            # validation lists not defined yet; skip
            pass

        # Plot loss graph (adversarial loss)
    plt.figure(1)
    for name in ['G_adv_loss_F', 'G_adv_loss_Q', 'D_adv_loss_F', 'D_adv_loss_Q']:
        loss_arr = torch.load(join(path_result, name + '.npy'), map_location='cpu')
        x_axis = np.arange(1, len(loss_arr) + 1)
        plt.plot(x_axis, loss_arr, label=name)
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.25))
    plt.legend(loc='upper right')
    plt.savefig(join(path_result, 'loss_curve_1.png'))
    plt.close() 
    
    # Plot loss graph (cycle consistency loss, identity loss)
    plt.figure(2)
    for name in ['G_cycle_loss_F', 'G_cycle_loss_Q', 'G_iden_loss_F', 'G_iden_loss_Q']:
        loss_arr = torch.load(join(path_result, name + '.npy'), map_location='cpu')
        plt.plot(x_axis, loss_arr, label=name)
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(join(path_result, 'loss_curve_2.png'))
    plt.close() 
    

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_checkpoint', type=str, default='./CT_denoising')
    parser.add_argument('--model_name', type=str, default='cyclegan_v1')
    parser.add_argument('--path_data', type=str, default='./AAPM_data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lambda_cycle', type=int, default=10)
    parser.add_argument('--lambda_iden', type=int, default=5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_epoch', type=int, default=12)
    parser.add_argument('--g_channels', type=int, default=32)
    parser.add_argument('--d_channels', type=int, default=64)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_checkpoint', action='store_true')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    train(
        path_checkpoint=args.path_checkpoint,
        model_name=args.model_name,
        path_data=args.path_data,
        batch_size=args.batch_size,
        lambda_cycle=args.lambda_cycle,
        lambda_iden=args.lambda_iden,
        beta1=args.beta1,
        beta2=args.beta2,
        num_epoch=args.num_epoch,
        g_channels=args.g_channels,
        d_channels=args.d_channels,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
        lr=args.lr,
        use_checkpoint=args.use_checkpoint
    )
