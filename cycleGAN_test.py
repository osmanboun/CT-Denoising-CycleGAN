import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json

from os import makedirs
from os.path import join, isdir
from tqdm.auto import tqdm
from cycleGAN_train import Generator, make_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import stats
from scipy import ndimage as ndi


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


# --- Additional image-quality metrics ---

def rmse_metric(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae_metric(a, b):
    return float(np.mean(np.abs(a - b)))


def nmse_metric(a, b):
    num = np.sum((a - b) ** 2)
    den = np.sum(b ** 2)
    return float(num / (den + 1e-12))


def hfen_metric(a, b, sigma=1.5):
    # High-Frequency Error Norm: RMSE of Laplacian of Gaussian filtered images
    log_a = ndi.gaussian_laplace(a, sigma=sigma)
    log_b = ndi.gaussian_laplace(b, sigma=sigma)
    return float(np.sqrt(np.mean((log_a - log_b) ** 2)))


def snr_metric(a, b):
    # SNR in dB: 20 * log10(||b|| / ||b - a||)
    num = np.linalg.norm(b)
    den = np.linalg.norm(b - a)
    if den == 0:
        return float('inf')
    return float(20.0 * np.log10((num + 1e-12) / (den + 1e-12)))


def compute_edge_errors_and_eki(input_img, output_img, ref_img):
    # Compute gradient magnitude (Sobel) for each image
    gx_in = ndi.sobel(input_img, axis=0)
    gy_in = ndi.sobel(input_img, axis=1)
    mag_in = np.hypot(gx_in, gy_in)

    gx_out = ndi.sobel(output_img, axis=0)
    gy_out = ndi.sobel(output_img, axis=1)
    mag_out = np.hypot(gx_out, gy_out)

    gx_ref = ndi.sobel(ref_img, axis=0)
    gy_ref = ndi.sobel(ref_img, axis=1)
    mag_ref = np.hypot(gx_ref, gy_ref)

    # L1 distances (edge errors)
    edge_err_input = np.sum(np.abs(mag_in - mag_ref))
    edge_err_output = np.sum(np.abs(mag_out - mag_ref))

    # EKI defined as 1 - (error_out / error_input)
    denom = edge_err_input + 1e-12
    eki = 1.0 - (edge_err_output / denom)
    return float(edge_err_input), float(edge_err_output), float(eki)


# --- Statistical helper functions ---

def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def rank_biserial(diff):
    diff = np.asarray(diff)
    n = len(diff)
    if n == 0:
        return 0.0
    absdiff = np.abs(diff)
    ranks = stats.rankdata(absdiff)
    pos = diff > 0
    neg = diff < 0
    r_pos = np.sum(ranks[pos])
    r_neg = np.sum(ranks[neg])
    denom = n * (n + 1) / 2.0
    return float((r_pos - r_neg) / denom)


def bootstrap_ci(data, statfunc=np.mean, n_boot=5000, ci=95, random_state=0):
    rng = np.random.RandomState(random_state)
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return (None, None)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boots.append(statfunc(sample))
    lower = np.percentile(boots, (100 - ci) / 2.0)
    upper = np.percentile(boots, 100 - (100 - ci) / 2.0)
    return float(lower), float(upper)


# --- Plotting helper functions ---

def ensure_dir(path):
    if not isdir(path):
        makedirs(path)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


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

    # Path for saving checkpoint
    if not isdir(path_checkpoint):
        makedirs(path_checkpoint)

    # Path for saving results
    path_result = join(path_checkpoint, model_name)
    ensure_dir(path_result)

    # Load test data with dataloader
    test_dataloader = make_dataloader(path_data, is_train=False)

    # Load the last checkpoint
    G_Q2F = Generator(1, 1, g_channels, ch_mult=ch_mult, num_res_blocks=num_res_blocks).to(device)
    checkpoint = torch.load(join(path_checkpoint, model_name + '.pth'), map_location=device)
    G_Q2F.load_state_dict(checkpoint['G_Q2F_state_dict'])
    G_Q2F.eval()

    # Run inference and save outputs
    with torch.no_grad():
        for _, x_Q, file_name in tqdm(test_dataloader):
            x_Q = x_Q.to(device)
            x_QF = G_Q2F(x_Q)[0].detach().cpu().numpy()
            x_QF = x_QF * 4000
            np.save(join(path_result, file_name[0]), x_QF[0])

    # Initialize lists for metrics
    psnr_quarter, ssim_quarter = [], []
    psnr_output, ssim_output = [], []
    rmse_input, mae_input, nmse_input = [], [], []
    rmse_output, mae_output, nmse_output = [], [], []
    hfen_input, hfen_output = [], []
    snr_input, snr_output = [], []
    edge_err_input_list, edge_err_output_list, eki_list = [], [], []
    filenames = []

    # 🔹 Automatically detect number of test files
    quarter_files = sorted(glob.glob(join(path_data, "test/quarter_dose", "*.npy")))
    full_files = sorted(glob.glob(join(path_data, "test/full_dose", "*.npy")))
    output_files = sorted(glob.glob(join(path_result, "*.npy")))

    num_files = min(len(quarter_files), len(full_files), len(output_files))
    print(f"Found {num_files} test files")

    if num_files == 0:
        print("No test files found. Exiting.")
        return

    # Calculate metrics
    for i in range(num_files):
        quarter = np.load(quarter_files[i])
        full = np.load(full_files[i])
        output = np.load(output_files[i])

        # Convert quarter/full to HU-like scaling as before
        quarter_s = (quarter - 0.0192) / 0.0192 * 1000
        full_s = (full - 0.0192) / 0.0192 * 1000
        output_s = output  # output already in expected scale when saved

        # Basic metrics (input vs GT and output vs GT)
        psnr_q = psnr(quarter_s, full_s)
        ssim_q = ssim(quarter_s, full_s)
        psnr_o = psnr(output_s, full_s)
        ssim_o = ssim(output_s, full_s)

        # RMSE/MAE/NMSE for input vs GT
        rmse_in = rmse_metric(quarter_s, full_s)
        mae_in = mae_metric(quarter_s, full_s)
        nmse_in = nmse_metric(quarter_s, full_s)

        # RMSE/MAE/NMSE for output vs GT
        rmse_out = rmse_metric(output_s, full_s)
        mae_out = mae_metric(output_s, full_s)
        nmse_out = nmse_metric(output_s, full_s)

        # HFEN for input and output
        hfen_in = hfen_metric(quarter_s, full_s, sigma=1.5)
        hfen_out = hfen_metric(output_s, full_s, sigma=1.5)

        # SNR for input and output
        snr_in = snr_metric(quarter_s, full_s)
        snr_out = snr_metric(output_s, full_s)

        # Edge errors and EKI (uses input, output and ref)
        edge_in, edge_out, eki_v = compute_edge_errors_and_eki(quarter_s, output_s, full_s)

        # Append
        psnr_quarter.append(psnr_q)
        ssim_quarter.append(ssim_q)
        psnr_output.append(psnr_o)
        ssim_output.append(ssim_o)

        rmse_input.append(rmse_in)
        mae_input.append(mae_in)
        nmse_input.append(nmse_in)
        hfen_input.append(hfen_in)
        snr_input.append(snr_in)

        rmse_output.append(rmse_out)
        mae_output.append(mae_out)
        nmse_output.append(nmse_out)
        hfen_output.append(hfen_out)
        snr_output.append(snr_out)

        edge_err_input_list.append(edge_in)
        edge_err_output_list.append(edge_out)
        eki_list.append(eki_v)

        filenames.append(quarter_files[i].split('/')[-1])

    print('Computed PSNR, SSIM, RMSE, MAE, NMSE, HFEN, SNR, edge errors, EKI for all slices')

    # --- Save per-slice metrics to CSV ---
    df = pd.DataFrame({
        'filename': filenames,
        # input vs GT
        'psnr_input': psnr_quarter,
        'ssim_input': ssim_quarter,
        'rmse_input': rmse_input,
        'mae_input': mae_input,
        'nmse_input': nmse_input,
        'hfen_input': hfen_input,
        'snr_input': snr_input,
        'edge_err_input': edge_err_input_list,
        # output vs GT
        'psnr_output': psnr_output,
        'ssim_output': ssim_output,
        'rmse_output': rmse_output,
        'mae_output': mae_output,
        'nmse_output': nmse_output,
        'hfen_output': hfen_output,
        'snr_output': snr_output,
        'edge_err_output': edge_err_output_list,
        # EKI (relative improvement of output vs input)
        'eki': eki_list
    })

    per_slice_csv = join(path_result, 'metrics_per_slice.csv')
    df.to_csv(per_slice_csv, index=False)

    # --- Create summary table: mean ± std, median + IQR, N ---
    def iqr(arr):
        q75, q25 = np.percentile(arr, [75, 25])
        return q75 - q25

    summary_rows = []
    metrics = {
        # input vs GT
        'psnr_input': psnr_quarter,
        'ssim_input': ssim_quarter,
        'rmse_input': rmse_input,
        'mae_input': mae_input,
        'nmse_input': nmse_input,
        'hfen_input': hfen_input,
        'snr_input': snr_input,
        'edge_err_input': edge_err_input_list,
        # output vs GT
        'psnr_output': psnr_output,
        'ssim_output': ssim_output,
        'rmse_output': rmse_output,
        'mae_output': mae_output,
        'nmse_output': nmse_output,
        'hfen_output': hfen_output,
        'snr_output': snr_output,
        'edge_err_output': edge_err_output_list,
        # EKI
        'eki': eki_list
    }

    for name, values in metrics.items():
        vals = np.array(values)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        median = float(np.median(vals))
        q75, q25 = np.percentile(vals, [75, 25])
        iqr_val = float(q75 - q25)
        N = int(len(vals))

        summary_rows.append({
            'metric': name,
            'mean': mean,
            'std': std,
            'mean±std': f"{mean:.3f} ± {std:.3f}",
            'median': median,
            'IQR': iqr_val,
            'median±IQR': f"{median:.3f} ± {iqr_val:.3f}",
            'N': N
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = join(path_result, 'summary_metrics.csv')
    summary_df.to_csv(summary_csv, index=False)

    # --- Print concise metric summary in the requested format ---
    def _mean_std(arr):
        a = np.array(arr)
        mean = float(np.mean(a)) if a.size > 0 else 0.0
        std = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        return mean, std

    m_psnr_in, s_psnr_in = _mean_std(psnr_quarter)
    m_psnr_out, s_psnr_out = _mean_std(psnr_output)
    m_ssim_in, s_ssim_in = _mean_std(ssim_quarter)
    m_ssim_out, s_ssim_out = _mean_std(ssim_output)
    m_rmse_in, s_rmse_in = _mean_std(rmse_input)
    m_rmse_out, s_rmse_out = _mean_std(rmse_output)
    m_mae_in, s_mae_in = _mean_std(mae_input)
    m_mae_out, s_mae_out = _mean_std(mae_output)
    m_nmse_in, s_nmse_in = _mean_std(nmse_input)
    m_nmse_out, s_nmse_out = _mean_std(nmse_output)
    m_hfen_in, s_hfen_in = _mean_std(hfen_input)
    m_hfen_out, s_hfen_out = _mean_std(hfen_output)
    m_snr_in, s_snr_in = _mean_std(snr_input)
    m_snr_out, s_snr_out = _mean_std(snr_output)
    # EKI: input baseline is 0 by definition in this script
    m_eki_out, s_eki_out = _mean_std(eki_list)

    print('\nMetric summary (mean ± std)')
    print(f"PSNR: input = {m_psnr_in:.3f} ± {s_psnr_in:.3f} | output = {m_psnr_out:.3f} ± {s_psnr_out:.3f}")
    print(f"SSIM: input = {m_ssim_in:.3f} ± {s_ssim_in:.3f} | output = {m_ssim_out:.3f} ± {s_ssim_out:.3f}")
    print(f"RMSE: input = {m_rmse_in:.3f} ± {s_rmse_in:.3f} | output = {m_rmse_out:.3f} ± {s_rmse_out:.3f}")
    print(f"MAE: input = {m_mae_in:.3f} ± {s_mae_in:.3f} | output = {m_mae_out:.3f} ± {s_mae_out:.3f}")
    print(f"NMSE: input = {m_nmse_in:.6f} ± {s_nmse_in:.6f} | output = {m_nmse_out:.6f} ± {s_nmse_out:.6f}")
    print(f"HFEN: input = {m_hfen_in:.3f} ± {s_hfen_in:.3f} | output = {m_hfen_out:.3f} ± {s_hfen_out:.3f}")
    print(f"SNR(dB): input = {m_snr_in:.3f} ± {s_snr_in:.3f} | output = {m_snr_out:.3f} ± {s_snr_out:.3f}")
    print(f"EKI: input = {0.0:.3f} ± {0.0:.3f} | output = {m_eki_out:.3f} ± {s_eki_out:.3f}")

    # --- Statistical tests on paired differences (PSNR & SSIM kept as before) ---
    stats_results = []

    comparisons = [
        ('psnr_output', 'psnr_input'),
        ('ssim_output', 'ssim_input')
    ]

    for a_name, b_name in comparisons:
        a = np.array(df[a_name])
        b = np.array(df[b_name])
        diff = a - b
        N = len(diff)

        # Shapiro-Wilk test for normality of differences (requires N>=3)
        try:
            sh_stat, sh_p = stats.shapiro(diff) if N >= 3 else (None, None)
        except Exception:
            sh_stat, sh_p = (None, None)

        normal = (sh_p is not None and sh_p > 0.05)

        # Choose appropriate paired test
        if normal:
            t_stat, t_p = stats.ttest_rel(a, b)
            test_used = 'paired t-test'
            test_stat = float(t_stat)
            test_p = float(t_p)
            # Cohen's d for paired samples
            eff_size = cohens_d_paired(a, b)
            eff_name = "Cohen's d"
        else:
            # Wilcoxon signed-rank test (two-sided)
            # If all differences are zero, scipy wilcoxon may error
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
                test_stat = float(w_stat)
                test_p = float(w_p)
            except Exception:
                test_stat, test_p = (None, None)
            test_used = 'wilcoxon signed-rank'
            eff_size = rank_biserial(diff)
            eff_name = 'rank-biserial'

        # Bootstrap 95% CI for mean difference
        ci_low, ci_high = bootstrap_ci(diff, statfunc=np.mean, n_boot=5000, ci=95, random_state=42)

        stats_results.append({
            'comparison': f"{a_name} vs {b_name}",
            'N': int(N),
            'shapiro_stat': float(sh_stat) if sh_stat is not None else None,
            'shapiro_p': float(sh_p) if sh_p is not None else None,
            'normal_diff': bool(normal),
            'test_used': test_used,
            'test_stat': test_stat,
            'test_p': test_p,
            'effect_name': eff_name,
            'effect_size': float(eff_size) if eff_size is not None else None,
            'mean_diff': float(np.mean(diff)) if N > 0 else None,
            'ci_mean_diff_95_low': ci_low,
            'ci_mean_diff_95_high': ci_high
        })

    # Save statistical test results
    stats_df = pd.DataFrame(stats_results)
    stats_csv = join(path_result, 'stat_tests.csv')
    stats_df.to_csv(stats_csv, index=False)

    stats_json = join(path_result, 'stat_tests.json')
    with open(stats_json, 'w') as fh:
        json.dump(stats_results, fh, indent=2)

    # --- Save run metadata ---
    metadata = {
        'path_checkpoint': path_checkpoint,
        'model_name': model_name,
        'checkpoint_file': join(path_checkpoint, model_name + '.pth'),
        'path_data': path_data,
        'g_channels': g_channels,
        'ch_mult': ch_mult,
        'num_res_blocks': num_res_blocks,
        'num_visualize': num_visualize,
        'seed': torch.initial_seed() if hasattr(torch, 'initial_seed') else None,
        'num_files': num_files
    }

    metadata_file = join(path_result, 'run_metadata.json')
    with open(metadata_file, 'w') as fh:
        json.dump(metadata, fh, indent=2)

    # ----------------------- VISUALIZATIONS -----------------------
    random.seed(int(metadata['seed']) if metadata['seed'] is not None else 0)
    sampled_indices = random.sample(range(num_files), min(num_visualize, num_files))

    # Prepare arrays for plotting
    psnr_q = np.array(psnr_quarter)
    psnr_o = np.array(psnr_output)
    ssim_q = np.array(ssim_quarter)
    ssim_o = np.array(ssim_output)

    # 1) Boxplots and violin plots for PSNR and SSIM
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    # PSNR boxplot
    axes1[0].boxplot([psnr_q, psnr_o], labels=['Input (Q)', 'Output'])
    axes1[0].set_title('PSNR: input vs output')
    axes1[0].set_ylabel('PSNR')
    # SSIM violin
    axes1[1].violinplot([ssim_q, ssim_o], showmeans=True)
    axes1[1].set_xticks([1, 2])
    axes1[1].set_xticklabels(['Input (Q)', 'Output'])
    axes1[1].set_title('SSIM: input vs output')
    save_figure(fig1, join(path_result, 'box_violin_psnr_ssim.png'))

    # 2) Scatter plots (input vs output) with identity line
    def scatter_with_identity(a, b, title, fname):
        mn = min(np.min(a), np.min(b))
        mx = max(np.max(a), np.max(b))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(a, b, alpha=0.6)
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.set_title(title)
        save_figure(fig, fname)

    scatter_with_identity(psnr_q, psnr_o, 'PSNR: input vs output', join(path_result, 'scatter_psnr.png'))
    scatter_with_identity(ssim_q, ssim_o, 'SSIM: input vs output', join(path_result, 'scatter_ssim.png'))

    # 3) Bland-Altman plots for bias analysis
    def bland_altman(a, b, title, fname):
        a = np.asarray(a)
        b = np.asarray(b)
        mean_ab = (a + b) / 2.0
        diff = b - a
        md = np.mean(diff)
        sd = np.std(diff, ddof=1)
        loa_upper = md + 1.96 * sd
        loa_lower = md - 1.96 * sd
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(mean_ab, diff, alpha=0.6)
        ax.axhline(md, color='k', linestyle='-')
        ax.axhline(loa_upper, color='r', linestyle='--')
        ax.axhline(loa_lower, color='r', linestyle='--')
        ax.set_xlabel('Mean of Input and Output')
        ax.set_ylabel('Output - Input')
        ax.set_title(f'{title}\nMD={md:.3f}, LoA=[{loa_lower:.3f}, {loa_upper:.3f}]')
        save_figure(fig, fname)

    bland_altman(psnr_q, psnr_o, 'Bland-Altman PSNR', join(path_result, 'bland_altman_psnr.png'))
    bland_altman(ssim_q, ssim_o, 'Bland-Altman SSIM', join(path_result, 'bland_altman_ssim.png'))

    # 4) Residual heatmaps (|output - full|) with consistent color scaling
    # Compute global vmax across sampled indices to keep consistent scaling
    max_resid = 0.0
    for i in range(num_files):
        full = np.load(full_files[i])
        output = np.load(output_files[i])
        resid = np.abs(output - full)
        max_resid = max(max_resid, np.nanmax(resid))

    resid_dir = join(path_result, 'residuals')
    ensure_dir(resid_dir)
    for idx in sampled_indices:
        quarter = np.load(quarter_files[idx])
        full = np.load(full_files[idx])
        output = np.load(output_files[idx])
        resid = np.abs(output - full)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(resid, vmin=0, vmax=max_resid)
        ax.set_title(f'Residual |output - full| ({filenames[idx]})')
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax)
        save_figure(fig, join(resid_dir, f'residual_{filenames[idx]}.png'))

    # 5) Qualitative multi-panel examples (quarter, full, output, diff)
    qual_dir = join(path_result, 'qual_examples')
    ensure_dir(qual_dir)
    for idx in sampled_indices:
        quarter = np.load(quarter_files[idx])
        full = np.load(full_files[idx])
        output = np.load(output_files[idx])
        # scale and clip for visualization
        quarter_v = np.clip((quarter - 0.0192) / 0.0192 * 1000, -1000, 1000)
        full_v = np.clip((full - 0.0192) / 0.0192 * 1000, -1000, 1000)
        output_v = np.clip(output, -1000, 1000)
        diff_v = np.abs(output_v - full_v)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(quarter_v, cmap='gray')
        axes[0].set_title('Quarter')
        axes[1].imshow(full_v, cmap='gray')
        axes[1].set_title('Full')
        axes[2].imshow(output_v, cmap='gray')
        axes[2].set_title('Output')
        im = axes[3].imshow(diff_v, vmin=0, vmax=np.max(diff_v))
        axes[3].set_title('|Output - Full|')
        for ax in axes:
            ax.axis('off')
        cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        save_figure(fig, join(qual_dir, f'qual_{filenames[idx]}.png'))

    # 6) CDF / histogram of metric improvements
    improvements_psnr = psnr_o - psnr_q
    improvements_ssim = ssim_o - ssim_q

    # Histogram + CDF for PSNR
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(improvements_psnr, bins=30, density=False, alpha=0.6)
    ax2 = ax.twinx()
    sorted_vals = np.sort(improvements_psnr)
    cdf = np.arange(1, len(sorted_vals) + 1) / float(len(sorted_vals))
    ax2.plot(sorted_vals, cdf, linestyle='-', linewidth=2)
    ax.set_xlabel('PSNR improvement (Output - Input)')
    ax.set_ylabel('Count')
    ax2.set_ylabel('CDF')
    save_figure(fig, join(path_result, 'hist_cdf_psnr.png'))

    # Histogram + CDF for SSIM
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(improvements_ssim, bins=30, density=False, alpha=0.6)
    ax2 = ax.twinx()
    sorted_vals = np.sort(improvements_ssim)
    cdf = np.arange(1, len(sorted_vals) + 1) / float(len(sorted_vals))
    ax2.plot(sorted_vals, cdf, linestyle='-', linewidth=2)
    ax.set_xlabel('SSIM improvement (Output - Input)')
    ax.set_ylabel('Count')
    ax2.set_ylabel('CDF')
    save_figure(fig, join(path_result, 'hist_cdf_ssim.png'))

    # 7) Paired-lines plot (input -> output improvement per slice)
    def paired_lines_plot(a, b, ylabel, fname, labels=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = [0, 1]
        for i in range(len(a)):
            ax.plot(x, [a[i], b[i]], marker='o', color='grey', alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Input', 'Output'])
        ax.set_ylabel(ylabel)
        ax.set_title(f'Paired-lines: {ylabel} (N={len(a)})')
        save_figure(fig, fname)

    paired_lines_plot(psnr_q, psnr_o, 'PSNR', join(path_result, 'paired_lines_psnr.png'))
    paired_lines_plot(ssim_q, ssim_o, 'SSIM', join(path_result, 'paired_lines_ssim.png'))

    print('Saved visualization figures to:', path_result)


if __name__ == '__main__':
    # Parse arguments
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

    # Set random seed
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
