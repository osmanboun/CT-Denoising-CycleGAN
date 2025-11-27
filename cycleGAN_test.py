import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob

from os import makedirs
from os.path import join, isdir
from tqdm.auto import tqdm
from cycleGAN_train import Generator, make_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Functions for calculating PSNR, SSIM
def psnr(A, ref):
    ref[ref < -1000] = -1000
    A[A < -1000] = -1000
    val_min = -1000
    val_max = np.amax(ref)
    ref = (ref - val_min) / (val_max - val_min)
    A = (A - val_min) / (val_max - val_min)
    return peak_signal_noise_ratio(ref, A)


def ssim(A, ref):
    ref[ref < -1000] = -1000
    A[A < -1000] = -1000
    val_min = -1000
    val_max = np.amax(ref)
    ref = (ref - val_min) / (val_max - val_min)
    A = (A - val_min) / (val_max - val_min)
    return structural_similarity(ref, A, data_range=2)


# Test function
def test(
    path_checkpoint='./CT_denoising',
    model_name='cyclegan_v1',
    path_data='./AAPM_data',
    g_channels=32,
    ch_mult=[1, 2, 4, 8],
    num_res_blocks=3,
    num_visualize=6
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not isdir(path_checkpoint):
        makedirs(path_checkpoint)

    path_result = join(path_checkpoint, model_name)
    if not isdir(path_result):
        makedirs(path_result)

    test_dataloader = make_dataloader(path_data, is_train=False)

    G_Q2F = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)
    checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'), map_location=device)
    G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
    G_Q2F.eval()

    with torch.no_grad():
        for _, x_Q, file_name in tqdm(test_dataloader):
            x_Q = x_Q.to(device)
            x_QF = G_Q2F(x_Q)[0].detach().cpu().numpy()
            x_QF = x_QF * 4000
            np.save(join(path_result, file_name[0]), x_QF[0])

    psnr_quarter, ssim_quarter = [], []
    psnr_output, ssim_output = [], []

    quarter_files = sorted(glob.glob(join(path_data, "test/quarter_dose", "*.npy")))
    full_files = sorted(glob.glob(join(path_data, "test/full_dose", "*.npy")))
    output_files = sorted(glob.glob(join(path_result, "*.npy")))

    num_files = min(len(quarter_files), len(full_files), len(output_files))
    print(f"Found {num_files} test files")

    for i in range(num_files):
        quarter = np.load(quarter_files[i], allow_pickle=True)
        full = np.load(full_files[i], allow_pickle=True)
        output = np.load(output_files[i], allow_pickle=True)

        quarter = (quarter - 0.0192) / 0.0192 * 1000
        full = (full - 0.0192) / 0.0192 * 1000

        psnr_quarter.append(psnr(quarter, full))
        ssim_quarter.append(ssim(quarter, full))
        psnr_output.append(psnr(output, full))
        ssim_output.append(ssim(output, full))

    print('PSNR and SSIM')
    print('Mean PSNR between input and ground truth:', np.mean(psnr_quarter))
    print('Mean SSIM between input and ground truth:', np.mean(ssim_quarter))
    print('Mean PSNR between network output and ground truth:', np.mean(psnr_output))
    print('Mean SSIM between network output and ground truth:', np.mean(ssim_output))

    plt.figure(figsize=(15, 30))
    sampled_indices = random.sample(range(num_files), num_visualize)

    for i, index in enumerate(sampled_indices):
        quarter = np.load(quarter_files[index], allow_pickle=True)
        full = np.load(full_files[index], allow_pickle=True)
        output = np.load(output_files[index], allow_pickle=True)

        quarter = np.clip((quarter - 0.0192) / 0.0192 * 1000, -1000, 1000)
        full = np.clip((full - 0.0192) / 0.0192 * 1000, -1000, 1000)
        output = np.clip(output, -1000, 1000)

        plt.subplot(num_visualize, 3, i * 3 + 1)
        plt.imshow(quarter, cmap='gray')
        plt.title('Quarter Dose', fontsize=16)
        plt.axis('off')

        plt.subplot(num_visualize, 3, i * 3 + 2)
        plt.imshow(full, cmap='gray')
        plt.title('Full Dose', fontsize=16)
        plt.axis('off')

        plt.subplot(num_visualize, 3, i * 3 + 3)
        plt.imshow(output, cmap='gray')
        plt.title('Output', fontsize=16)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(join(path_result, 'qualitative_cyclegan.png'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_checkpoint', type=str, default='./CT_denoising')
    parser.add_argument('--model_name', type=str, default='cyclegan_v1')
    parser.add_argument('--path_data', type=str, default='./AAPM_data')
    parser.add_argument('--g_channels', type=int, default=32)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_visualize', type=int, default=6)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    test(
        path_checkpoint=args.path_checkpoint,
        model_name=args.model_name,
        path_data=args.path_data,
        g_channels=args.g_channels,
        ch_mult=args.ch_mult,
        num_res_blocks=args.num_res_blocks,
        num_visualize=args.num_visualize
    )
