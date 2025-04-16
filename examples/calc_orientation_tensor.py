import argparse
import os
import torch
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

# Adjust these imports to your local structure
from datasets.metric_colmap import Parser, Dataset
from gsplat.rendering import rasterization


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


def get_camera_dir(camtoworld: torch.Tensor) -> torch.Tensor:
    """
    Extracts a single representative direction for the camera,
    e.g. the negative z-axis from the camtoworld matrix [4,4].
    Returns a 3D float tensor on the same device.
    """
    # local camera Z-axis in world space is camtoworld[:3,2]
    # If we define the 'view direction' as the negative of that axis:
    cam_z = camtoworld[0, :3, 2]  # shape [3], pick the batch=0
    view_dir = -cam_z
    return view_dir / (view_dir.norm() + 1e-8)


def direction_to_bin(dir_vec: torch.Tensor, bin_count: int) -> int:
    """
    Maps a single direction vector (3D) to an integer bin [0..bin_count-1].
    We'll do a simple lat-lon partition, requiring bin_count to be a perfect square.
    """
    dir_vec = dir_vec / (dir_vec.norm() + 1e-8)
    x, y, z = dir_vec[0].item(), dir_vec[1].item(), dir_vec[2].item()

    theta = np.arccos(max(min(z, 1.0), -1.0))  # [0, pi]
    phi = np.arctan2(y, x)                     # [-pi, pi)
    if phi < 0:
        phi += 2.0 * np.pi  # make [0, 2pi)

    # assume bin_count = L*L
    L = int(np.sqrt(bin_count))
    if L * L != bin_count:
        raise ValueError(f"bin_count={bin_count} is not a perfect square.")

    lat_bins = L
    lon_bins = L

    frac_theta = theta / np.pi            # [0,1]
    frac_phi   = phi   / (2.0 * np.pi)    # [0,1]

    i = int(frac_theta * lat_bins)
    j = int(frac_phi   * lon_bins)

    i = min(i, lat_bins - 1)
    j = min(j, lon_bins - 1)

    bin_idx = i * lon_bins + j
    return bin_idx


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Render Gaussians from training views and store coverage bins."
    )
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--factor", type=int, default=1, help="downsample factor for the dataset")
    parser.add_argument("--bin_count", type=int, default=64, help="number of direction bins per Gaussian")
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

    # 2) Load Gaussians
    gaussians = load_ply(args.ply_path)
    device = "cuda"

    means = gaussians.xyz
    scales = torch.exp(gaussians.scaling)
    quats = gaussians.rotation
    opacities = torch.sigmoid(gaussians.opacity).squeeze(-1)

    if gaussians.features_rest is not None:
        colors = torch.cat([gaussians.features_dc, gaussians.features_rest], dim=1)
    else:
        colors = gaussians.features_dc

    means  = means.to(device)
    scales = scales.to(device)
    quats  = quats.to(device)
    opacities = opacities.to(device)
    colors = colors.to(device)

    N_gauss = len(opacities)

    # 3) We'll keep a single float "weights" array for the entire set,
    #    plus a coverage bins array with shape [N_gauss, bin_count].
    coverage_bins = torch.zeros(N_gauss, args.bin_count, device=device)

    print(f"Processing {len(trainloader)} images to fill coverage bins...")
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), desc="Rendering Progress"):
        weights = torch.zeros_like(opacities)  # shape [N_gauss]
        camtoworld = data["camtoworld"].to(device)  # shape [B,4,4], B=1
        Ks = data["K"].to(device)                   # shape [B,3,3], B=1
        images = data["image"]                      # shape [B,H,W,3] or [B,3,H,W]

        # Convert images to float [0..1], though we might not need them actually
        if images.dtype == torch.uint8:
            images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        images = images.to(device)

        # Extract height/width
        if images.ndim == 4 and images.shape[-1] == 3:
            # => [1,H,W,3]
            H, W = images.shape[1], images.shape[2]
        else:
            # => [1,3,H,W]
            _, _, H, W = images.shape

        # 3.1) Rasterize to get "weights" updated
        #     The rasterization must implement something like 'weights[g] = max( weights[g], alpha_g )'
        #     or accumulative, depending on your code. 
        #     We'll assume your kernel updates 'weights' in-place.
        rendered_batch, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld),  # shape [B,4,4]
            Ks=Ks,                                  # shape [B,3,3]
            width=W,
            height=H,
            sh_degree=gaussians.active_sh_degree,
            near_plane=0.01,
            far_plane=1e10,
            camera_model="pinhole",
            packed=False,
            sparse_grad=False,
            weights=weights,         # <--- used/updated by the CUDA kernel
            rasterize_mode="classic",
        )

        # 3.2) Get a single "camera direction" for this view
        # We use the negative Z approach
        view_dir = get_camera_dir(camtoworld)  # shape [1,3]

        # We'll take the first (and only) from the batch
        # 3.3) Convert that direction to a bin
        bin_idx = direction_to_bin(view_dir, args.bin_count)

        # 3.4) Accumulate the newly updated 'weights' to coverage_bins
        coverage_bins[:, bin_idx] = torch.max(weights, coverage_bins[:, bin_idx])

        # 3.5) Reset or not? 
        # If you want to keep max alpha across *all* views in 'weights',
        # you might not reset. If you want per-view, you can zero it out:
        # weights.zero_()

    print("\nFinished coverage bins accumulation!")
    # coverage_bins shape => [N_gauss, bin_count]

    # Optionally, save coverage_bins to disk
    coverage_bins_cpu = coverage_bins.cpu()
    coverage_path = os.path.join(args.out_dir, "coverage_bins.pt")
    torch.save(coverage_bins_cpu, coverage_path)
    print(f"Saved coverage bins to: {coverage_path}")

    # Also save the final 'weights' if you want
    weights_path = os.path.join(args.out_dir, "final_weights.pt")
    torch.save(weights.cpu(), weights_path)
    print(f"Saved final weights to: {weights_path}")

    print("Done!")


if __name__ == "__main__":
    main()
