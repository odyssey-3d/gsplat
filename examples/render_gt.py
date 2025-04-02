import argparse
import os
import torch
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from fused_ssim import fused_ssim
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.colmap import Parser, Dataset
from gsplat.rendering import rasterization
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from scipy.stats import bootstrap


@dataclass
class GaussianData:
    active_sh_degree: int
    xyz: torch.Tensor  # (N, 3)
    features_dc: torch.Tensor  # (N, 1, 3) if DC or colors
    features_rest: Optional[torch.Tensor]  # (N, M, 3) if higher-order SH, else None
    scaling: torch.Tensor  # (N, 3)
    rotation: torch.Tensor  # (N, 4) quaternions
    opacity: torch.Tensor  # (N,)


def load_ply(filepath: str) -> GaussianData:
    """
    Loads a PLY file and constructs a GaussianData object with correctly shaped tensors.
    Ensures compatibility with rendering.
    """
    print(f"Loading PLY from {filepath}")
    pcd = o3d.t.io.read_point_cloud(filepath)

    # **Positions**
    xyz = torch.tensor(pcd.point.positions.numpy(), dtype=torch.float32, device="cuda")

    # **Scales**
    scales = np.column_stack([pcd.point[f"scale_{i}"].numpy() for i in range(3)])
    scaling = torch.tensor(scales, dtype=torch.float32, device="cuda")

    # **Rotations (Quaternions)**
    rotations = np.column_stack([pcd.point[f"rot_{i}"].numpy() for i in range(4)])
    rotation = torch.tensor(rotations, dtype=torch.float32, device="cuda")

    # **Opacity**
    opacity = torch.tensor(
        pcd.point["opacity"].numpy(), dtype=torch.float32, device="cuda"
    ).squeeze(-1)

    # **Prepare Outputs**
    active_sh_degree = 3  # default or adjust if needed

    # **Spherical Harmonic Coefficients** (DC + higher orders)
    if "f_dc_0" in pcd.point:
        # Count and validate DC coefficients
        dc_count = sum(1 for key in pcd.point if key.startswith("f_dc_"))
        if dc_count != 3:
            raise ValueError(f"Expected exactly 3 f_dc_ attributes, found {dc_count}")

        # Load DC coefficients: shape (N, 3)
        sh0 = np.column_stack([pcd.point[f"f_dc_{i}"].numpy() for i in range(dc_count)])

        # Check for higher-order SH coefficients
        rest_count = sum(1 for key in pcd.point if key.startswith("f_rest_"))
        if rest_count > 0:
            if rest_count % 3 != 0:
                raise ValueError(
                    f"Number of f_rest_ attributes must be multiple of 3, found {rest_count}"
                )
            M = rest_count // 3

            # Load rest coefficients: shape (N, 3*M)
            shN = np.column_stack(
                [pcd.point[f"f_rest_{i}"].numpy() for i in range(rest_count)]
            )
            shN_tensor = torch.tensor(shN, dtype=torch.float32, device="cuda")

            # Reshape to (N, M, 3)
            features_rest = torch.stack(
                (
                    shN_tensor[:, :M],
                    shN_tensor[:, M : 2 * M],
                    shN_tensor[:, 2 * M : 3 * M],
                ),
                dim=2,
            )
            # DC terms remain SH coefficients
            features_dc = (
                torch.tensor(sh0, dtype=torch.float32, device="cuda").unsqueeze(1)
            )  # (N,1,3)
        else:
            # No higher-order SH => interpret DC as color in SH(0) => invert
            color_inverted = sh0 * 0.2820947917738781 + 0.5
            features_dc = (
                torch.tensor(color_inverted, dtype=torch.float32, device="cuda")
                .unsqueeze(1)
            )
            features_rest = None
    else:
        raise ValueError("No f_dc_ attributes found in the PLY file")

    return GaussianData(
        active_sh_degree=active_sh_degree,
        xyz=xyz,
        features_dc=features_dc,
        features_rest=features_rest,
        scaling=scaling,
        rotation=rotation,
        opacity=opacity,
    )


def compute_median_confidence_interval(data, confidence=0.95, n_resamples=10000):
    """
    Computes the 2-sided confidence interval for the median using a bootstrap method.
    Returns (ci_low, ci_high).
    """
    data = np.array(data, dtype=float)  # ensure float
    res = bootstrap(
        (data,),
        np.median,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method='BCa',
        random_state=123
    )
    return res.confidence_interval.low, res.confidence_interval.high


def plot_metric_separately(name, values, out_dir, strikes=None):
    """
    Plots the given metric in two subplots:
      1) A line plot over image index (with statistics + 95% CI for the median).
      2) A histogram of values.

    Optionally highlights 'strikes' as vertical spans.

    - Draws a median line on the top subplot.
    - Shows stats (mean, median, std, var, min, max).
    - Adds a shaded region for the 95% confidence interval of the median.
    """
    mean_val = values.mean()
    median_val = np.median(values)
    std_val = values.std()
    var_val = values.var()
    min_val = values.min()
    max_val = values.max()

    ci_low, ci_high = compute_median_confidence_interval(values, confidence=0.95)

    indices = np.arange(len(values))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(indices, values, marker='o', label=name, color='blue', alpha=0.7)
    ax1.axhline(y=median_val, color='red', linestyle='--', label="Median")
    ax1.axhspan(ci_low, ci_high, color='orange', alpha=0.2, label="95% CI (Median)")

    if strikes is not None:
        for (start_idx, end_idx) in strikes:
            ax1.axvspan(start_idx, end_idx, color='yellow', alpha=0.1)

    ax1.set_title(f"{name} vs. Image Index")
    ax1.set_xlabel("Image Index")
    ax1.set_ylabel(name)
    ax1.legend(loc="best")

    stats_str = "\n".join([
        f"Median:  {median_val:.4f}",
        f"Mean:    {mean_val:.4f}",
        f"Std:     {std_val:.4f}",
        f"Var:     {var_val:.4f}",
        f"Min:     {min_val:.4f}",
        f"Max:     {max_val:.4f}",
        f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]"
    ])
    ax1.text(
        0.05,
        0.95,
        stats_str,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax2.hist(values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_title(f"{name} Distribution")
    ax2.set_xlabel(f"{name} Value")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name.lower()}_analysis.png"))
    plt.close()


def find_long_strikes(psnr_array, ssim_array, lpips_array,
                      psnr_med_low, ssim_med_low, lpips_med_high):
    """
    A "long strike" is a consecutive set of frames for which:
      - PSNR >= psnr_med_low
      - SSIM >= ssim_med_low
      - LPIPS <= lpips_med_high
    """
    n = len(psnr_array)
    if n == 0:
        return []

    good_mask = (
        (psnr_array >= psnr_med_low) &
        (ssim_array >= ssim_med_low) &
        (lpips_array <= lpips_med_high)
    )

    strikes = []
    strike_start = None
    for i in range(n):
        if good_mask[i]:
            if strike_start is None:
                strike_start = i
        else:
            if strike_start is not None:
                strikes.append((strike_start, i - 1))
                strike_start = None
    if strike_start is not None:
        strikes.append((strike_start, n - 1))

    # Sort by length descending
    strikes.sort(key=lambda x: (x[1] - x[0] + 1), reverse=True)
    return strikes


def create_strike_stripe(out_dir, strike_index, start, end):
    """
    Reads each 'compare_{i:04d}.png' in [start,end], horizontally concatenates,
    and saves 'strike_{strike_index}_{start}_{end}.png'.
    """
    images = []
    for i in range(start, end + 1):
        compare_path = os.path.join(out_dir, f"compare_{i:04d}.png")
        if not os.path.exists(compare_path):
            continue
        img = cv2.imread(compare_path)  # Use cv2 instead of imageio
        images.append(img)

    if len(images) == 0:
        return

    stripe = np.concatenate(images, axis=1)
    strike_filename = f"strike_{strike_index}_{start}_{end}.png"
    out_path = os.path.join(out_dir, strike_filename)
    cv2.imwrite(out_path, stripe)  # Use cv2 instead of imageio


def main():
    parser = argparse.ArgumentParser(
        description="Render Gaussians vs. GT images using a DataLoader in parallel."
    )
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_comparisons", action="store_true", default=False,
                        help="Save side-by-side comparison images (default: False)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --------------------------------------------------
    # 1) Prepare dataset & dataloader
    # --------------------------------------------------
    print("Creating dataset/dataloader with your snippet:")
    colmap_parser = Parser(
        data_dir=args.colmap_dir,
        factor=args.factor,
        normalize=False,
        test_every=-1,
    )
    trainset = Dataset(colmap_parser, split="train")

    trainloader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )
    trainloader_iter = iter(trainloader)

    # --------------------------------------------------
    # 2) Load your Gaussians from PLY
    # --------------------------------------------------
    gaussians = load_ply(args.ply_path)
    device = "cuda"
    means = gaussians.xyz.to(device)
    scales = torch.exp(gaussians.scaling).to(device)
    quats = gaussians.rotation.to(device)
    opacities = torch.sigmoid(gaussians.opacity).to(device).squeeze(-1)

    if gaussians.features_rest is not None:
        colors = torch.cat([gaussians.features_dc, gaussians.features_rest], dim=1).to(device)
    else:
        colors = gaussians.features_dc.to(device)

    # --------------------------------------------------
    # 3) Set up metrics & accumulators
    # --------------------------------------------------
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # We'll gather them in lists
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    # --------------------------------------------------
    # 4) Main loop with progress bar
    # --------------------------------------------------
    print("Processing images...")
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), desc="Rendering Progress"):
        camtoworld = data["camtoworld"].to(device)  # [B,4,4]
        Ks = data["K"].to(device)                   # [B,3,3]
        images = data["image"]                      # [B, H, W, 3] or [B,3,H,W]
        # We'll do a minimal example, assume batch_size=1 for illustration
        # If batch_size>1, you'd loop or do everything in parallel.

        # Convert images to float [0..1]
        if images.dtype == torch.uint8:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        images = images.to(device)

        # Suppose batch_size=1 for clarity
        # shape => [1, H, W, 3] or [1,3,H,W]
        if images.ndim == 4 and images.shape[-1] == 3:
            # => [1,H,W,3]
            H, W = images.shape[1], images.shape[2]
            gt_img = images  # [1,H,W,3]
            # make sure shape is consistent
        else:
            # => [1,3,H,W]
            _, _, H, W = images.shape
            # might need to permute if your dataset is different
            gt_img = images.permute(0, 2, 3, 1)  # => [1,H,W,3]

        # Now do the rendering
        with torch.no_grad():
            rendered_batch, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworld),  # [B,4,4]
                Ks=Ks,                                  # [B,3,3]
                width=W,
                height=H,
                sh_degree=3,
                near_plane=0.01,
                far_plane=1e10,
                camera_model="pinhole",
                packed=False,
                sparse_grad=False,
                rasterize_mode="classic",
            )

        # rendered_batch => [B, H, W, 3]
        # If B=1, shape => [1, H, W, 3]
        rendered = rendered_batch[0].clamp(0.0, 1.0)  # => [H, W, 3]

        # Prepare for metric calc
        # Must be => [1,3,H,W]
        gt_torch = gt_img.permute(0, 3, 1, 2)  # => [1,3,H,W]
        rend_torch = rendered.unsqueeze(0).permute(0, 3, 1, 2).to(device)  # => [1,3,H,W]

        # Metrics
        psnr_val = psnr_metric(rend_torch, gt_torch).item()
        ssim_val_tensor = fused_ssim(rend_torch, gt_torch)
        if isinstance(ssim_val_tensor, torch.Tensor):
            ssim_val = ssim_val_tensor.item()
        else:
            ssim_val = float(ssim_val_tensor)
        lpips_val = lpips_metric(rend_torch, gt_torch).item()

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

        # Optional side-by-side comparison
        if args.save_comparisons:
            # Convert rendered image to numpy uint8
            rendered_cpu = rendered.cpu().numpy()  # [H, W, 3], float [0..1]
            rendered_np_255 = (rendered_cpu * 255).astype(np.uint8)

            # Convert ground truth image to numpy uint8
            gt_img_cpu = gt_img.squeeze(0).cpu()  # [H, W, 3] or adjust if needed
            gt_np_255 = (gt_img_cpu.numpy() * 255).astype(np.uint8)

            # Concatenate horizontally (GT | Rendered)
            compare_img = np.concatenate((gt_np_255, rendered_np_255), axis=1)
            compare_path = os.path.join(args.out_dir, f"compare_{i:04d}.png")
            cv2.imwrite(compare_path, cv2.cvtColor(compare_img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    # ------------------------------------------------------------
    # 5) After loop: compute final stats & plots
    # ------------------------------------------------------------
    psnr_array = np.array(psnr_vals)
    ssim_array = np.array(ssim_vals)
    lpips_array = np.array(lpips_vals)
    n = len(psnr_array)
    indices = np.arange(n)

    if n == 0:
        print("No images processed, done!")
        return

    # Median-based stats
    psnr_median = np.median(psnr_array)
    ssim_median = np.median(ssim_array)
    lpips_median = np.median(lpips_array)

    # 95% CI for each
    psnr_med_low, psnr_med_high = compute_median_confidence_interval(psnr_array)
    ssim_med_low, ssim_med_high = compute_median_confidence_interval(ssim_array)
    lpips_med_low, lpips_med_high = compute_median_confidence_interval(lpips_array)

    # Also compute mean/std
    psnr_mean = psnr_array.mean()
    psnr_std = psnr_array.std()
    ssim_mean = ssim_array.mean()
    ssim_std = ssim_array.std()
    lpips_mean = lpips_array.mean()
    lpips_std = lpips_array.std()

    # Identify "strikes"
    strikes = find_long_strikes(
        psnr_array,
        ssim_array,
        lpips_array,
        psnr_med_low,
        ssim_med_low,
        lpips_med_high
    )

    # Plot combined metrics
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.plot(indices, psnr_array, label="PSNR", color='blue')
    ax.axhline(y=psnr_median, color='blue', linestyle='--', label="PSNR Median")
    ax.axhspan(psnr_med_low, psnr_med_high, color='blue', alpha=0.05)

    ax.plot(indices, ssim_array, label="SSIM", color='green')
    ax.axhline(y=ssim_median, color='green', linestyle='--', label="SSIM Median")
    ax.axhspan(ssim_med_low, ssim_med_high, color='green', alpha=0.05)

    ax.plot(indices, lpips_array, label="LPIPS", color='red')
    ax.axhline(y=lpips_median, color='red', linestyle='--', label="LPIPS Median")
    ax.axhspan(lpips_med_low, lpips_med_high, color='red', alpha=0.05)

    for (start_idx, end_idx) in strikes:
        ax.axvspan(start_idx, end_idx, color='yellow', alpha=0.1)

    ax.set_xlabel("Step Index")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics Evolution (Median + 95% CI)")
    ax.legend(loc="best")

    stats_str = (
        f"PSNR: median={psnr_median:.4f}, CI=[{psnr_med_low:.4f}, {psnr_med_high:.4f}], "
        f"mean={psnr_mean:.4f}, std={psnr_std:.4f}\n"
        f"SSIM: median={ssim_median:.4f}, CI=[{ssim_med_low:.4f}, {ssim_med_high:.4f}], "
        f"mean={ssim_mean:.4f}, std={ssim_std:.4f}\n"
        f"LPIPS: median={lpips_median:.4f}, CI=[{lpips_med_low:.4f}, {lpips_med_high:.4f}], "
        f"mean={lpips_mean:.4f}, std={lpips_std:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_str,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "metrics_evolution.png"))
    plt.close()

    # Separate plots for each metric
    plot_metric_separately("PSNR", psnr_array, args.out_dir, strikes=strikes)
    plot_metric_separately("SSIM", ssim_array, args.out_dir, strikes=strikes)
    plot_metric_separately("LPIPS", lpips_array, args.out_dir, strikes=strikes)

    # ------------------------------------------------------------
    # 9) Print the longest strikes and create stripes
    # ------------------------------------------------------------
    print("\nLongest strikes (ordered by descending length):")
    top_strikes = strikes[:5]  # print/make stripes for top 5
    for idx, (start, end) in enumerate(top_strikes):
        length = end - start + 1
        print(f"  From frame {start} to frame {end} (length = {length})")

    for idx, (start, end) in enumerate(top_strikes):
        create_strike_stripe(args.out_dir, idx, start, end)

    # ------------------------------------------------------------
    # 10) Create a JSON summary report
    # ------------------------------------------------------------
    # Per-image metrics
    per_image_metrics = []
    for i in range(n):
        per_image_metrics.append({
            "index": i,
            "psnr": float(psnr_array[i]),
            "ssim": float(ssim_array[i]),
            "lpips": float(lpips_array[i])
        })

    # Full strikes info
    strikes_list = []
    for (start_idx, end_idx) in strikes:
        strikes_list.append({
            "start_idx": start_idx,
            "end_idx": end_idx
        })

    # Build the final JSON structure
    json_report = {
        "per_image": per_image_metrics,
        "summary_statistics": {
            "psnr": {
                "median": float(psnr_median),
                "ci_low": float(psnr_med_low),
                "ci_high": float(psnr_med_high),
                "mean": float(psnr_mean),
                "std": float(psnr_std)
            },
            "ssim": {
                "median": float(ssim_median),
                "ci_low": float(ssim_med_low),
                "ci_high": float(ssim_med_high),
                "mean": float(ssim_mean),
                "std": float(ssim_std)
            },
            "lpips": {
                "median": float(lpips_median),
                "ci_low": float(lpips_med_low),
                "ci_high": float(lpips_med_high),
                "mean": float(lpips_mean),
                "std": float(lpips_std)
            }
        },
        "strikes": strikes_list
    }

    # Write out the JSON report
    json_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    print("\nJSON summary report saved to:", json_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
