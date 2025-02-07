from concurrent.futures import ProcessPoolExecutor
import os
import json
from typing import Dict, Tuple
from pathlib import Path
from collections import defaultdict
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from PIL import Image
from gsplat.rendering import rasterization
import open3d as o3d
import numpy as np
import torch
from torch import Tensor
import tqdm as tqdm
from lib_bilagrid import color_correct
from datasets.colmap import Dataset, Parser


def load_ply(filepath: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load a PLY file and extract all its attributes into a dictionary of torch tensors.
    """
    print(f"Loading PLY from {filepath}")
    pcd = o3d.t.io.read_point_cloud(filepath)

    # Initialize dictionary for all attributes
    data = {}

    # Extract point positions (means)
    means = pcd.point.positions.numpy()
    data["means"] = torch.from_numpy(means).to(device)

    # Extract scales
    scales = np.column_stack([
        pcd.point[f"scale_{i}"].numpy() for i in range(3)
    ])
    data["scales"] = torch.from_numpy(scales).to(device)

    # Extract quaternions
    quats = np.column_stack([
        pcd.point[f"rot_{i}"].numpy() for i in range(4)
    ])
    data["quats"] = torch.from_numpy(quats).to(device)

    # Extract opacities
    data["opacities"] = torch.from_numpy(pcd.point["opacity"].numpy()).to(device)

    # Check if we have SH coefficients or colors
    if "f_dc_0" in pcd.point:
        # Count number of SH coefficients
        dc_count = sum(1 for key in pcd.point if key.startswith("f_dc_"))
        rest_count = sum(1 for key in pcd.point if key.startswith("f_rest_"))

        if rest_count > 0:  # We have SH coefficients
            # Extract SH0 coefficients
            sh0 = np.column_stack([
                pcd.point[f"f_dc_{i}"].numpy() for i in range(dc_count)
            ])
            data["sh0"] = torch.from_numpy(sh0).reshape(-1, dc_count // 3, 3)

            # Extract SHN coefficients
            shN = np.column_stack([
                pcd.point[f"f_rest_{i}"].numpy() for i in range(rest_count)
            ])
            data["shN"] = torch.from_numpy(shN).reshape(-1, rest_count // 3, 3)
        else:  # We have colors
            colors = np.column_stack([
                pcd.point[f"f_dc_{i}"].numpy() for i in range(dc_count)
            ])
            data["colors"] = torch.from_numpy(colors * 0.2820947917738781 + 0.5)

    data["opacities"] = data["opacities"].squeeze(-1)
    data["colors"] = torch.cat([data["sh0"], data["shN"]], dim=1).to(device)
    return data

@torch.no_grad()
def rasterize_splats(
        splats,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    means = splats["means"]  # [N, 3]

    # rasterization does normalization internally!!
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]

    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        **kwargs,
    )
    return render_colors, render_alphas, info


def merge_checkpoints(ckpts_folder: str):
    ckpts_folder = Path(ckpts_folder)
    if not ckpts_folder.exists():
        raise ValueError(f"Checkpoint folder does not exist: {ckpts_folder}")

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
        ckpts = [torch.load(f, map_location='cuda') for f in files]

        # Create a new ParameterDict to store merged parameters
        merged_splats = torch.nn.ParameterDict()

        # Get the keys from first checkpoint's splats
        splat_keys = ckpts[0]['splats'].keys()

        # Merge parameters for each key
        for key in splat_keys:
            merged_data = torch.cat([ckpt['splats'][key] for ckpt in ckpts])
            merged_splats[key] = torch.nn.Parameter(merged_data)

        # Save merged data as PLY
    return merged_splats

def evaluate_frame(data, plys, device, metrics, ssim, psnr, lpips):
    camtoworlds = data["camtoworld"].to(device)
    Ks = data["K"].to(device)
    pixels = data["image"].to(device) / 255.0
    height, width = pixels.shape[1:3]

    renders, _, _ = rasterize_splats(
        splats=plys,
        camtoworlds=camtoworlds,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=3,
        render_mode="RGB",
    )
    colors = renders[..., 0:3] if renders.shape[-1] == 4 else renders

    pixels_p = pixels.permute(0, 3, 1, 2)
    colors_p = colors.permute(0, 3, 1, 2)
    cc_colors = color_correct(colors, pixels)
    cc_colors_p = cc_colors.permute(0, 3, 1, 2)

    frame_metrics = {
        "cc_psnr": psnr(cc_colors_p, pixels_p).item(),
        "cc_ssim": ssim(cc_colors_p, pixels_p).item(),
        "cc_lpips": lpips(cc_colors_p, pixels_p).item(),
        "psnr": psnr(colors_p, pixels_p).item(),
        "ssim": ssim(colors_p, pixels_p).item(),
        "lpips": lpips(colors_p, pixels_p).item()
    }

    return frame_metrics, colors.squeeze(0).cpu().numpy().clip(0, 1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--eval-dir", type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir), f"Data directory {args.data_dir} does not exist."
    assert os.path.exists(args.ply), f"PLY file {args.ply} does not exist."
    assert os.path.exists(args.eval_dir), f"Eval directory {args.eval_dir} does not exist."
    assert os.path.exists(args.ckpt), f"Checkpoint file {args.ckpt} does not exist."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plys = merge_checkpoints(args.ckpt)
    #means = plys["means"].to(device)
    #quats = plys["quats"].to(device)
    #scales = plys["scales"].to(device)
    #opacities = plys["opacities"].to(device)
    #sh0 = plys["sh0"].to(device)
    #shN = plys["shN"].to(device)

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, training=False
    )
    eval_dataset  = Dataset(parser, split="val", load_depths=False)
    print(f"Dataset: {len(eval_dataset)} images.")

    # Load PLY file.
    #splats = load_ply(args.ply, device=device)

    # debug equal data
    #assert torch.allclose(splats["means"], means)
    #assert torch.allclose(splats["quats"], quats)
    #assert torch.allclose(splats["scales"], scales)
    #assert torch.allclose(splats["opacities"], opacities)
    #assert torch.allclose(splats["sh0"].to(device), sh0)
    #assert torch.allclose(splats["shN"].to(device), shN)

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    data_iter = iter(dataloader)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)

    metrics = defaultdict(list)
    all_renders = []

    pbar = tqdm.tqdm(range(len(eval_dataset)), desc="Rendering and evaluating")
    for step in pbar:
        data = next(data_iter)
        frame_metrics, render = evaluate_frame(data, plys, device, metrics, ssim, psnr, lpips)

        for k, v in frame_metrics.items():
            metrics[k].append(v)
        all_renders.append(render)

    # Save metrics
    stats = {k: np.mean(v) for k, v in metrics.items()}
    with open(f"{args.eval_dir}/metrics.json", "w") as f:
        json.dump(stats, f, indent=2)


    # Save images in parallel
    def save_image(args):
        idx, render, path = args
        Image.fromarray((render * 255).astype(np.uint8)).save(
            f"{path}/{idx:03d}.png", optimize=False)


    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        save_args = [(i, render, args.eval_dir) for i, render in enumerate(all_renders)]
        list(tqdm.tqdm(executor.map(save_image, save_args), total=len(save_args), desc="Saving images"))
