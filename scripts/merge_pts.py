import torch
import argparse
import open3d as o3d
from pathlib import Path
from collections import defaultdict
import numpy as np


def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    print(f"Saving ply to {dir}")

    # Convert all tensors to numpy arrays
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    # Extract data arrays
    means = numpy_data["means"]
    scales = numpy_data["scales"]
    quats = numpy_data["quats"]
    opacities = numpy_data["opacities"]

    # Remove outliers based on position
    mean_pos = np.mean(means, axis=0)
    distances = np.linalg.norm(means - mean_pos, axis=1)
    std_dist = np.std(distances)
    inliers = distances <= 4 * std_dist  # Points within 4 standard deviations

    # Filter all data arrays
    means = means[inliers]
    scales = scales[inliers]
    quats = quats[inliers]
    opacities = opacities[inliers]

    # Handle colors or spherical harmonics based on whether colors is provided
    if colors is not None:
        colors = colors.detach().cpu().numpy()[inliers]
    else:
        sh0 = numpy_data["sh0"][inliers].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()
        shN = numpy_data["shN"][inliers].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()

    # Initialize ply_data with positions and normals
    ply_data = {
        "positions": o3d.core.Tensor(means, dtype=o3d.core.Dtype.Float32),
        "normals": o3d.core.Tensor(np.zeros_like(means), dtype=o3d.core.Dtype.Float32),
    }

    # Add features
    if colors is not None:
        # Use provided colors, converted to SH coefficients
        for j in range(colors.shape[1]):
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                (colors[:, j: j + 1] - 0.5) / 0.2820947917738781,
                dtype=o3d.core.Dtype.Float32,
            )
    else:
        # Use spherical harmonics (sh0 for DC, shN for rest)
        for j in range(sh0.shape[1]):
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                sh0[:, j: j + 1], dtype=o3d.core.Dtype.Float32
            )
        for j in range(shN.shape[1]):
            ply_data[f"f_rest_{j}"] = o3d.core.Tensor(
                shN[:, j: j + 1], dtype=o3d.core.Dtype.Float32
            )

    # Add opacity
    ply_data["opacity"] = o3d.core.Tensor(
        opacities.reshape(-1, 1), dtype=o3d.core.Dtype.Float32
    )

    # Add scales
    for i in range(scales.shape[1]):
        ply_data[f"scale_{i}"] = o3d.core.Tensor(
            scales[:, i: i + 1], dtype=o3d.core.Dtype.Float32
        )

    # Add rotations
    for i in range(quats.shape[1]):
        ply_data[f"rot_{i}"] = o3d.core.Tensor(
            quats[:, i: i + 1], dtype=o3d.core.Dtype.Float32
        )

    # Create and save the point cloud
    pcd = o3d.t.geometry.PointCloud(ply_data)
    success = o3d.t.io.write_point_cloud(dir, pcd)
    assert success, "Could not save ply file."


def merge_checkpoints(ckpts_folder: str, output_dir: str = None):
    """
    Load and merge checkpoint files from a folder into PLY files, filtering out invalid Gaussians.

    Args:
        ckpts_folder: Folder containing checkpoint files
        output_dir: Output directory for PLY files. If None, uses ckpts_folder parent
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

    # Group checkpoints by step number
    step_groups = defaultdict(list)
    for ckpt_file in ckpt_files:
        # Extract step number from filename (assumes format ckpt_STEP_rank*.pt)
        step = int(ckpt_file.name.split('_')[1])
        step_groups[step].append(ckpt_file)

    print(f"Found checkpoints for {len(step_groups)} steps: {sorted(step_groups.keys())}")

    # Process each step
    for step, files in sorted(step_groups.items()):
        print(f"\nProcessing step {step}...")
        print(f"Found {len(files)} rank files:")
        for f in files:
            print(f"  {f}")

        # Load all checkpoints for this step
        ckpts = [torch.load(f, map_location='cpu') for f in files]

        # Create a new ParameterDict to store merged parameters
        merged_splats = torch.nn.ParameterDict()

        # Get the keys from first checkpoint's splats
        splat_keys = ckpts[0]['splats'].keys()

        # Merge parameters for each key
        for key in splat_keys:
            merged_data = torch.cat([ckpt['splats'][key] for ckpt in ckpts])
            merged_splats[key] = torch.nn.Parameter(merged_data)

        # Compute valid mask to filter out NaN or Inf values
        num_gaussians = merged_splats['means'].shape[0]
        valid_mask = torch.ones(num_gaussians, dtype=torch.bool, device=merged_splats['means'].device)
        for attr in merged_splats.values():
            valid_mask = valid_mask & torch.isfinite(attr).view(attr.shape[0], -1).all(dim=1)

        # Filter the splats to keep only valid Gaussians
        filtered_splats = torch.nn.ParameterDict({
            key: torch.nn.Parameter(attr[valid_mask]) for key, attr in merged_splats.items()
        })

        # Log the number of filtered Gaussians
        num_filtered = valid_mask.sum().item()
        print(f"Filtered out {num_gaussians - num_filtered} invalid Gaussians out of {num_gaussians}")

        # Save merged data as PLY
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