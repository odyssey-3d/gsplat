import argparse
import os
import torch
import numpy as np
import open3d as o3d
import imageio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fused_ssim import fused_ssim
from typing import Optional
from tqdm import tqdm  # for the progress bar

# ----------------------------------------------------------------
# Adjust these imports to match your environment:
from datasets.colmap import Parser, Dataset
from gsplat.rendering import rasterization

# TorchMetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
# ----------------------------------------------------------------

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
            )  # (N,1,3)
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

def plot_metric_separately(name, values, out_dir):
    """
    Plots the given metric in two subplots:
      1) A line plot over image index (with statistics).
      2) A histogram of values.

    - Draws a mean line on the top subplot.
    - Shows stats (mean, median, std, var, min, max).
    """
    mean_val = values.mean()
    median_val = np.median(values)
    std_val = values.std()
    var_val = values.var()
    min_val = values.min()
    max_val = values.max()

    indices = np.arange(len(values))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot main metric vs index
    ax1.plot(indices, values, marker='o', label=name)
    # Draw horizontal mean line
    ax1.axhline(y=mean_val, label="Mean")

    ax1.set_title(f"{name} vs. Image Index")
    ax1.set_xlabel("Image Index")
    ax1.set_ylabel(name)
    ax1.legend()

    # Build a string for stats
    stats_str = "\n".join([
        f"Mean:   {mean_val:.4f}",
        f"Median: {median_val:.4f}",
        f"Std:    {std_val:.4f}",
        f"Var:    {var_val:.4f}",
        f"Min:    {min_val:.4f}",
        f"Max:    {max_val:.4f}",
    ])
    # Place stats in the top-left corner of the upper subplot
    ax1.text(
        0.05,
        0.95,
        stats_str,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Histogram
    ax2.hist(values, bins=20, alpha=0.7)
    ax2.set_title(f"{name} Distribution")
    ax2.set_xlabel(f"{name} Value")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name.lower()}_analysis.png"))
    plt.close()

def find_stable_runs(psnr, ssim, lpips, psnr_thresh=0.1, ssim_thresh=0.01, lpips_thresh=0.01):
    """
    Find the runs (start_index, end_index) of consecutive frames where
    the absolute changes in each metric between consecutive frames do not
    exceed the given thresholds. Each run is inclusive of start and end indices.
    """
    stable_runs = []
    if len(psnr) == 0:
        return stable_runs

    start_idx = 0
    for i in range(len(psnr) - 1):
        # Check changes for the three metrics from frame i to i+1
        if (abs(psnr[i+1] - psnr[i]) > psnr_thresh or
            abs(ssim[i+1] - ssim[i]) > ssim_thresh or
            abs(lpips[i+1] - lpips[i]) > lpips_thresh):
            # We have exceeded the threshold => end the current stable run
            stable_runs.append((start_idx, i))
            start_idx = i + 1

    # Don't forget the final run
    stable_runs.append((start_idx, len(psnr) - 1))

    # Sort runs by descending length
    stable_runs.sort(key=lambda r: (r[1] - r[0] + 1), reverse=True)

    return stable_runs

def main():
    parser = argparse.ArgumentParser(
        description="Render (via Gaussians) and compare with GT images. Save only the side-by-side comparisons."
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        required=True,
        help="Path to the COLMAP data (images, sparse, etc.).",
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        required=True,
        help="Path to the .ply file of your Gaussian splats.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Folder where to save comparison images and plots.",
    )
    parser.add_argument(
        "--factor", type=int, default=1, help="Downsample factor for loading dataset."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load the COLMAP dataset (training split)
    colmap_parser = Parser(
        data_dir=args.colmap_dir,
        factor=args.factor,
        normalize=False,  # Must match your training usage
        test_every=-1,   # put all images into "train" for demonstration
    )
    trainset = Dataset(colmap_parser, split="train")

    # 2) Load the Gaussians from .ply
    gaussians = load_ply(args.ply_path)
    device = "cuda"

    # Convert log-params to actual
    means = gaussians.xyz.to(device)
    scales = torch.exp(gaussians.scaling).to(device)
    quats = gaussians.rotation.to(device)
    opacities = torch.sigmoid(gaussians.opacity).to(device).squeeze(-1)

    # Merge DC + higher-order features
    if gaussians.features_rest is not None:
        colors = torch.cat([gaussians.features_dc, gaussians.features_rest], dim=1).to(device)
    else:
        colors = gaussians.features_dc.to(device)

    # Initialize metrics
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # Lists to store metrics
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    # 3) Render each training image, compute metrics, and save side-by-side
    for i in tqdm(range(len(trainset)), desc="Rendering & Saving Comparisons"):
        example = trainset[i]

        camtoworld = example["camtoworld"].to(device)  # [4,4]
        K = example["K"].to(device)                    # [3,3]

        gt_img = example["image"]  # shape [H, W, 3] or [3, H, W]
        # Ensure GT is float in [0..1]
        if gt_img.dtype == torch.uint8:
            gt_img = gt_img.float()
        if gt_img.max() > 1.0:
            gt_img = gt_img / 255.0

        # Figure out shape (H, W)
        if gt_img.ndim == 3 and gt_img.shape[-1] == 3:
            H, W = gt_img.shape[:2]  # [H, W, 3]
        elif gt_img.ndim == 3 and gt_img.shape[0] == 3:
            H, W = gt_img.shape[1], gt_img.shape[2]  # [3, H, W]
        else:
            raise ValueError(f"Unexpected GT shape: {gt_img.shape}")

        # Render via rasterization
        with torch.no_grad():
            rendered, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworld)[None],
                Ks=K[None],
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
        # rendered => [1, H, W, 3]
        render_img = rendered[0].clamp(0.0, 1.0)  # [H, W, 3]
        render_img_cpu = render_img.cpu()

        # Prepare GT for metrics: [1, 3, H, W]
        if gt_img.ndim == 3 and gt_img.shape[-1] == 3:
            gt_torch = gt_img.permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            gt_torch = gt_img.unsqueeze(0).to(device)

        # Prepare render for metrics: [1, 3, H, W]
        rendered_torch = render_img_cpu.permute(2, 0, 1).unsqueeze(0).to(device)

        # Safety check
        if gt_torch.max() > 1.0 or rendered_torch.max() > 1.0:
            raise ValueError(
                "Either GT or Render is above 1.0. Ensure both are normalized to [0..1].\n"
                f"GT range: [{gt_torch.min()}, {gt_torch.max()}], "
                f"Render range: [{rendered_torch.min()}, {rendered_torch.max()}]"
            )

        # Compute metrics as scalars (floats)
        psnr_val = psnr_metric(rendered_torch, gt_torch).item()
        ssim_val_tensor = fused_ssim(rendered_torch, gt_torch)
        if isinstance(ssim_val_tensor, torch.Tensor):
            ssim_val = ssim_val_tensor.item()
        else:
            ssim_val = float(ssim_val_tensor)
        lpips_val = lpips_metric(rendered_torch, gt_torch).item()

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

        # Create side-by-side (compare) image: [H, W*2, 3] in [0..255]
        render_img_np = render_img_cpu.numpy()  # [H, W, 3], float [0..1]
        render_img_np_255 = (render_img_np * 255).astype(np.uint8)

        gt_img_cpu = gt_img.cpu()
        if gt_img_cpu.ndim == 3 and gt_img_cpu.shape[0] == 3:
            gt_img_cpu = gt_img_cpu.permute(1, 2, 0)
        gt_img_np = gt_img_cpu.numpy()  # [H, W, 3], float [0..1]
        gt_img_np_255 = (gt_img_np * 255).astype(np.uint8)

        compare_img = np.concatenate((gt_img_np_255, render_img_np_255), axis=1)
        compare_path = os.path.join(args.out_dir, f"compare_{i:04d}.png")
        imageio.imwrite(compare_path, compare_img)

    # ------------------------------------------------------------
    # 4) Plot combined metrics evolution (PSNR, SSIM, LPIPS)
    # ------------------------------------------------------------
    indices = np.arange(len(trainset))
    psnr_array = np.array(psnr_vals)
    ssim_array = np.array(ssim_vals)
    lpips_array = np.array(lpips_vals)

    # Compute stats for each metric
    psnr_mean, psnr_median = psnr_array.mean(), np.median(psnr_array)
    psnr_std, psnr_var = psnr_array.std(), psnr_array.var()

    ssim_mean, ssim_median = ssim_array.mean(), np.median(ssim_array)
    ssim_std, ssim_var = ssim_array.std(), ssim_array.var()

    lpips_mean, lpips_median = lpips_array.mean(), np.median(lpips_array)
    lpips_std, lpips_var = lpips_array.std(), lpips_array.var()

    plt.figure(figsize=(10, 6))
    plt.plot(indices, psnr_array, label="PSNR")
    plt.plot(indices, ssim_array, label="SSIM")
    plt.plot(indices, lpips_array, label="LPIPS")
    plt.xlabel("Image Index")
    plt.ylabel("Metric Value")
    plt.title("Metrics Evolution over Training Images")
    plt.legend()

    # Add a text box with summary stats
    text_stats = (
        f"PSNR: mean={psnr_mean:.4f}, median={psnr_median:.4f}, std={psnr_std:.4f}, var={psnr_var:.4f}\n"
        f"SSIM: mean={ssim_mean:.4f}, median={ssim_median:.4f}, std={ssim_std:.4f}, var={ssim_var:.4f}\n"
        f"LPIPS: mean={lpips_mean:.4f}, median={lpips_median:.4f}, std={lpips_std:.4f}, var={lpips_var:.4f}"
    )
    ax = plt.gca()
    ax.text(
        0.02,
        0.98,
        text_stats,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "metrics_evolution.png"))
    plt.close()

    # ------------------------------------------------------------
    # 5) Create separate plots with statistics (+ mean line)
    # ------------------------------------------------------------
    plot_metric_separately("PSNR", psnr_array, args.out_dir)
    plot_metric_separately("SSIM", ssim_array, args.out_dir)
    plot_metric_separately("LPIPS", lpips_array, args.out_dir)

    # ------------------------------------------------------------
    # 6) Identify and print the longest stable runs
    # ------------------------------------------------------------
    # Adjust these thresholds as needed
    psnr_thresh = 0.1
    ssim_thresh = 0.01
    lpips_thresh = 0.01

    stable_runs = find_stable_runs(psnr_array, ssim_array, lpips_array,
                                   psnr_thresh=psnr_thresh,
                                   ssim_thresh=ssim_thresh,
                                   lpips_thresh=lpips_thresh)

    print("\nStable runs (longest first):")
    # Print the top 5 runs, or all if fewer than 5
    for (start, end) in stable_runs[:5]:
        length = end - start + 1
        print(f"  From {start} to {end} (length = {length})")

    print("\nDone!")

if __name__ == "__main__":
    main()
