import torch
import argparse
import open3d as o3d
from pathlib import Path
from collections import defaultdict
import numpy as np


def save_ply(
        splats: torch.nn.ParameterDict,
        path: str,
        colors: torch.Tensor = None,
        scene_scale: float = 1.0,
):
    """
    Save PLY file from the parameter dictionary `splats`.
    This version handles homogeneous coordinates (log w)
    and converts them to standard 3D for both positions and scales
    *in world space*, applying `scene_scale` if requested.
    Opacities are also sigmoid'ed to store the final alpha.
    """
    print(f"Saving ply to {path}")

    # Convert all tensors to numpy arrays
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    # Required keys
    means = numpy_data["means"]               # (N, 3) in "M = X * W" form if 'w' is present
    scales_log = numpy_data["scales"]         # (N, 3) in log space
    quats = numpy_data["quats"]               # (N, 4)
    opacities_logit = numpy_data["opacities"] # (N,) or (N,1), logit of alpha

    # Check if we have 'w' (log of W) in the dictionary
    has_w = "w" in numpy_data
    if has_w:
        w_log = numpy_data["w"]              # (N,)
        w = np.exp(w_log).reshape(-1, 1)     # W in linear space
        # Convert homogeneous means -> 3D means in world space
        means_3d = (means / w)
    else:
        # If there's no 'w', we assume 'means' is already standard 3D
        means_3d = means

    # Convert logit(opacities) -> alpha in [0..1]
    opacities = 1.0 / (1.0 + np.exp(-opacities_logit))  # shape (N,)

    # Convert scales from log space into real 3D scale, consistent with the
    # training-time logic. If 'w' is present, we do the same as the rasterizer:
    #    scale_3d = exp(scales_log) * (1 / w) * norm(M)
    if has_w:
        norms_means = np.linalg.norm(means, axis=1, keepdims=True)  # (N, 1)
        w = np.exp(w_log).reshape(-1, 1)     # W in linear space
        scales_3d = np.exp(scales_log) / w * norms_means
    else:
        # If no w, we assume 'scales' is raw log(\sigma)
        scales_3d = np.exp(scales_log)

    means_3d *= scene_scale
    scales_3d *= scene_scale

    mean_pos = np.mean(means_3d, axis=0)
    distances = np.linalg.norm(means_3d - mean_pos, axis=1)
    std_dist = np.std(distances)
    inliers = distances <= 6 * std_dist  # keep everything within 6 sigma
    # Filter all arrays by inliers
    means_3d = means_3d[inliers]
    scales_3d = np.log(scales_3d[inliers])
    quats = quats[inliers]
    opacities = opacities[inliers]

    # Handle colors or spherical harmonics
    if colors is not None:
        colors = colors.detach().cpu().numpy()[inliers]
    else:
        sh0 = numpy_data["sh0"][inliers].transpose(0, 2, 1).reshape(means_3d.shape[0], -1).copy()
        shN = numpy_data["shN"][inliers].transpose(0, 2, 1).reshape(means_3d.shape[0], -1).copy()

    # Initialize ply_data
    ply_data = {
        "positions": o3d.core.Tensor(means_3d, dtype=o3d.core.Dtype.Float32),
        "normals": o3d.core.Tensor(np.zeros_like(means_3d), dtype=o3d.core.Dtype.Float32),
    }

    # Add features
    if colors is not None:
        # Use provided colors, converted to SH-like DC
        for j in range(colors.shape[1]):
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                (colors[:, j : j + 1] - 0.5) / 0.2820947917738781,
                dtype=o3d.core.Dtype.Float32,
                )
    else:
        # Use spherical harmonics
        for j in range(sh0.shape[1]):
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                sh0[:, j : j + 1], dtype=o3d.core.Dtype.Float32
            )
        for j in range(shN.shape[1]):
            ply_data[f"f_rest_{j}"] = o3d.core.Tensor(
                shN[:, j : j + 1], dtype=o3d.core.Dtype.Float32
            )

    # Add opacity
    ply_data["opacity"] = o3d.core.Tensor(
        opacities.reshape(-1, 1), dtype=o3d.core.Dtype.Float32
    )

    # Add scales
    for i in range(scales_3d.shape[1]):
        ply_data[f"scale_{i}"] = o3d.core.Tensor(
            scales_3d[:, i : i + 1], dtype=o3d.core.Dtype.Float32
        )

    # Add rotations
    for i in range(quats.shape[1]):
        ply_data[f"rot_{i}"] = o3d.core.Tensor(
            quats[:, i : i + 1], dtype=o3d.core.Dtype.Float32
        )

    # Create point cloud
    pcd = o3d.t.geometry.PointCloud(ply_data)
    success = o3d.t.io.write_point_cloud(dir, pcd)
    assert success, "Could not save ply file."


def merge_checkpoints(ckpts_folder: str, output_dir: str = None):
    """
    Load and merge checkpoint files from a folder into PLY files, filtering out invalid Gaussians.
    """
    ckpts_folder = Path(ckpts_folder)
    if not ckpts_folder.exists():
        raise ValueError(f"Checkpoint folder does not exist: {ckpts_folder}")

    # If no output directory specified, create a 'merged_ply' folder next to ckpts
    if output_dir is None:
        output_dir = ckpts_folder.parent / "merged_ply"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all checkpoint files
    ckpt_files = list(ckpts_folder.glob("ckpt_*_rank*.pt"))
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in: {ckpts_folder}")

    # Group by step
    step_groups = defaultdict(list)
    for ckpt_file in ckpt_files:
        # Format assumed: ckpt_STEP_rankX.pt
        step = int(ckpt_file.name.split('_')[1])
        step_groups[step].append(ckpt_file)

    print(f"Found checkpoints for {len(step_groups)} steps: {sorted(step_groups.keys())}")

    # Process each step
    for step, files in sorted(step_groups.items()):
        print(f"\nProcessing step {step}...")
        print(f"Found {len(files)} rank files:")
        for f in files:
            print(f"  {f}")

        # Load all checkpoints
        ckpts = [torch.load(f, map_location='cpu') for f in files]

        # Create a new ParameterDict to store merged parameters
        merged_splats = torch.nn.ParameterDict()

        # Merge data for each key
        for key in ckpts[0]['splats'].keys():
            merged_data = torch.cat([ckpt['splats'][key] for ckpt in ckpts])
            merged_splats[key] = torch.nn.Parameter(merged_data)

        # Filter out any Gaussians that are NaN or Inf
        num_gaussians = merged_splats['means'].shape[0]
        valid_mask = torch.ones(num_gaussians, dtype=torch.bool)
        for attr in merged_splats.values():
            valid_mask = valid_mask & torch.isfinite(attr).view(attr.shape[0], -1).all(dim=1)

        # Apply mask
        filtered_splats = torch.nn.ParameterDict({
            key: torch.nn.Parameter(attr[valid_mask]) for key, attr in merged_splats.items()
        })

        num_filtered = valid_mask.sum().item()
        print(f"Filtered out {num_gaussians - num_filtered} invalid Gaussians out of {num_gaussians}")

        # Save to .ply
        output_ply = output_dir / f"gaussian_step_{step}.ply"
        save_ply(filtered_splats, str(output_ply))
        print(f"Successfully saved merged PLY to {output_ply}")

def main():
    parser = argparse.ArgumentParser(description='Merge checkpoint files from a folder into PLY files')
    parser.add_argument('--ckpts-folder', type=str, required=True,
                        help='Folder containing checkpoint files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for PLY files (optional)')

    args = parser.parse_args()
    merge_checkpoints(args.ckpts_folder, args.output_dir)

if __name__ == '__main__':
    main()
