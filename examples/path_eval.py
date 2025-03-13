import os
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_ellipse_path_y,
    generate_spiral_path,
)
import imageio
import open3d as o3d
import numpy as np
import torch
import tyro
from datasets.colmap import Dataset, Parser
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import rgb_to_sh, set_random_seed
from gsplat.rendering import rasterization
import matplotlib.pyplot as plt


# Existing GaussianData and load_ply
@dataclass
class GaussianData:
    means: Tensor
    opacities: Tensor
    scales: Tensor
    quats: Tensor
    sh0: Tensor
    shN: Tensor


def load_ply(path: str) -> GaussianData:
    """
    Load a PLY file and extract Gaussian Splatting parameters
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    print(f"Loading PLY file: {path}")
    pcd = o3d.t.io.read_point_cloud(path)

    # Get points (means)
    means = torch.from_numpy(pcd.point.positions.numpy()).float()
    num_points = means.shape[0]

    print(f"Loaded {num_points} points from PLY file")

    # Initialize parameter dictionaries
    params = {}

    # Extract opacities
    if 'opacity' in pcd.point:
        opacities = torch.from_numpy(pcd.point.opacity.numpy()).float()
    else:
        print("WARNING: No opacities found, using default value 1.0")
        opacities = torch.ones((num_points, 1))

    # Extract scales
    scales = torch.ones((num_points, 3))
    for i in range(3):
        scale_key = f'scale_{i}'
        if scale_key in pcd.point:
            scales[:, i] = torch.from_numpy(pcd.point[scale_key].numpy()).float().squeeze()

    # Extract rotations/quaternions
    quats = torch.zeros((num_points, 4))
    quats[:, 0] = 1.0  # Default to identity rotation
    for i in range(4):
        rot_key = f'rot_{i}'
        if rot_key in pcd.point:
            quats[:, i] = torch.from_numpy(pcd.point[rot_key].numpy()).float().squeeze()

    # Extract SH coefficients
    # First, check for DC components (f_dc_X)
    sh0 = torch.zeros((num_points, 1, 3))
    for i in range(3):
        dc_key = f'f_dc_{i}'
        if dc_key in pcd.point:
            sh0[:, 0, i] = torch.from_numpy(pcd.point[dc_key].numpy()).float().squeeze()

    # Check for higher-order components (f_rest_X)
    # Count how many rest components we have
    rest_components = 0
    for i in range(100):  # Arbitrary large number to check
        if f'f_rest_{i}' in pcd.point:
            rest_components += 1
        else:
            break

    # If no SH components found, use colors directly
    if rest_components == 0 and not any(f'f_dc_{i}' in pcd.point for i in range(3)):
        if 'colors' in pcd.point:
            print("No SH coefficients found, using colors")
            rgb = torch.from_numpy(pcd.point.colors.numpy()).float()
            sh0[:, 0, :] = rgb_to_sh(rgb)

    # Extract higher-order SH coefficients
    if rest_components > 0:
        shN = torch.zeros((num_points, rest_components // 3, 3))
        for i in range(rest_components):
            rest_key = f'f_rest_{i}'
            if rest_key in pcd.point:
                comp_idx = i // 3
                color_idx = i % 3
                shN[:, comp_idx, color_idx] = torch.from_numpy(pcd.point[rest_key].numpy()).float().squeeze()
    else:
        # If no higher-order components, create empty tensor
        shN = torch.zeros((num_points, 0, 3))

    print(f"Prepared tensors with shapes: means={means.shape}, sh0={sh0.shape}, shN={shN.shape}")
    return GaussianData(means=means, opacities=opacities, scales=scales, quats=quats, sh0=sh0, shN=shN)


# Config with Trajectory Parameters
@dataclass
class Config:
    data_dir: str = ""
    eval_dir: str = "eval"
    model_path: str = ""
    sh_degree: int = 3
    white_bkgd: bool = False
    near_plane: float = 0.01
    far_plane: float = 1000.0
    batch_size: int = 1
    num_steps: int = 1
    eval_step: int = 1
    num_val: int = -1
    path_type: str = "none"  # Options: "none", "spiral", "ellipse_z", "ellipse_y", "interpolated"
    n_frames: int = 120
    n_rots: int = 2
    zrate: float = 0.5
    spiral_scale_f: float = 1.0
    spiral_scale_r: float = 1.0
    focus_distance: float = 0.75
    variation: float = 0.0
    phase: float = 0.0
    height: float = 0.0
    n_interp: int = 10
    spline_degree: int = 5
    smoothness: float = 0.03
    rot_weight: float = 0.1


# Runner Class with updated render_path
class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.world_rank = 0  # Assuming single GPU
        self.writer = SummaryWriter(log_dir=cfg.eval_dir)
        self.render_dir = os.path.join(cfg.eval_dir, "renders")
        os.makedirs(self.render_dir, exist_ok=True)

        # Create parser
        self.parser = Parser(data_dir=cfg.data_dir, test_every=-1)

        # Create dataset
        self.valset = Dataset(self.parser, "train")
        print(f"Created validation dataset with {len(self.valset)} images")

        # Load Gaussian data from PLY
        data = load_ply(cfg.model_path)
        # Store in format matching simple_trainer.py
        self.splats = {
            "means": data.means.to(self.device),
            "quats": data.quats.to(self.device),
            "scales": torch.log(data.scales).to(self.device),  # Store scales in log space
            "opacities": torch.logit(data.opacities.squeeze(-1)).to(self.device),  # Store opacities in logit space
            "sh0": data.sh0.to(self.device),
            "shN": data.shN.to(self.device)
        }

        # Metrics
        self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)

        # Print model summary
        print(f"Model loaded: {len(self.splats['means'])} Gaussians")
        print(f"SH coefficients: sh0={self.splats['sh0'].shape}, shN={self.splats['shN'].shape}")

    def rasterize_splats(self, camtoworlds: Tensor, Ks: Tensor, width: int, height: int,
                         sh_degree: int, near_plane: float, far_plane: float, masks: Optional[Tensor] = None) -> Tuple[
        Tensor, Tensor, Dict]:
        """
        Rasterize 3D Gaussians to produce rendered images - following simple_trainer.py approach
        """
        means = self.splats["means"]
        quats = self.splats["quats"]  # rasterization does normalization internally
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        # Combine SH coefficients as in simple_trainer.py
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        # Ensure camera matrices are well-formed
        try:
            # Check if camera matrices are valid
            if torch.any(torch.isnan(camtoworlds)) or torch.any(torch.isinf(camtoworlds)):
                print("WARNING: Camera matrices contain NaN or Inf values")
                # Clean up NaN and Inf values
                camtoworlds = torch.nan_to_num(camtoworlds, nan=0.0, posinf=1e6, neginf=-1e6)

            # Compute view matrices as inverse of camera-to-world matrices
            viewmats = torch.linalg.inv(camtoworlds)
        except Exception as e:
            print(f"ERROR inverting camera matrices: {e}")
            print(f"Camera matrix shape: {camtoworlds.shape}")
            print(f"Camera matrix:\n{camtoworlds}")

            # Use a fallback method if inversion fails
            viewmats = torch.zeros_like(camtoworlds)
            for i in range(camtoworlds.shape[0]):
                try:
                    # Try to correct the matrix before inversion
                    cam_fixed = camtoworlds[i].clone()
                    # Ensure the rotation part is orthogonal
                    U, _, V = torch.linalg.svd(cam_fixed[:3, :3], full_matrices=True)
                    cam_fixed[:3, :3] = U @ V
                    viewmats[i] = torch.linalg.inv(cam_fixed)
                except:
                    # Last resort - create an identity view matrix
                    print(f"WARNING: Using identity view matrix for camera {i}")
                    viewmats[i] = torch.eye(4, device=camtoworlds.device)
                    viewmats[i, :3, 3] = -camtoworlds[i, :3, 3]  # Approximately correct translation

        # Run rasterization
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=False,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def eval(self, step: int):
        # Handle num_val correctly to limit the number of samples
        max_samples = min(len(self.valset), self.cfg.num_val) if self.cfg.num_val > 0 else len(self.valset)
        losses = defaultdict(list)

        for i in range(max_samples):
            data = self.valset[i]
            camtoworld = data["camtoworld"].to(self.device).unsqueeze(0)
            K = data["K"].to(self.device).unsqueeze(0)

            # Get the original image dimensions
            image = data["image"].to(self.device)
            width, height = image.shape[1], image.shape[0]

            # Create ground truth in the correct format [B, C, H, W]
            gt_rgb = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            masks = data["mask"].to(self.device) if "mask" in data else None

            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworld,
                Ks=K,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                masks=masks,
            )
            colors = torch.clamp(colors, 0.0, 1.0)  # [1, H, W, 3]

            if self.world_rank == 0:
                # Debug shape information
                print(f"Rendered colors shape: {colors.shape}, GT RGB shape: {gt_rgb.shape}")

                # Convert to numpy for saving
                if len(colors.shape) == 4:  # [B, H, W, C]
                    canvas = colors[0].cpu().numpy()  # Use indexing instead of squeeze to be explicit
                else:
                    print(f"WARNING: Unexpected colors shape: {colors.shape}")
                    continue

                print(f"Canvas shape: {canvas.shape}")

                # Convert to uint8 for saving
                canvas = (canvas * 255).astype(np.uint8)

                # Save image
                imageio.imwrite(f"{self.render_dir}/val_{i:04d}.png", canvas)

                # Calculate metrics - convert colors to match gt_rgb format [B, C, H, W]
                colors_for_metrics = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]

                # Make sure the height and width are in the right order (not swapped)
                # gt_rgb should be in [B, C, H, W] format and colors_for_metrics also [B, C, H, W]
                if colors_for_metrics.shape[2:] != gt_rgb.shape[2:]:
                    print(
                        f"WARNING: Shape mismatch (H/W swapped): colors={colors_for_metrics.shape}, gt_rgb={gt_rgb.shape}")
                    if colors_for_metrics.shape[2] == gt_rgb.shape[3] and colors_for_metrics.shape[3] == gt_rgb.shape[
                        2]:
                        # Height and width are swapped in ground truth, transpose them
                        print("Transposing ground truth to match rendered image")
                        gt_rgb = gt_rgb.transpose(2, 3)
                    else:
                        # Different dimensions, resize
                        print("Resizing to match dimensions")
                        import torch.nn.functional as F
                        gt_rgb = F.interpolate(gt_rgb, size=colors_for_metrics.shape[2:], mode='bilinear',
                                               align_corners=False)

                # Calculate metrics
                psnr = self.psnr_fn(colors_for_metrics, gt_rgb).item()
                ssim = self.ssim_fn(colors_for_metrics, gt_rgb).item()
                lpips = self.lpips_fn(colors_for_metrics, gt_rgb).item()
                losses["psnr"].append(psnr)
                losses["ssim"].append(ssim)
                losses["lpips"].append(lpips)

        if self.world_rank == 0 and losses:
            for k, v in losses.items():
                self.writer.add_scalar(f"val/{k}", np.mean(v), step)
            print(
                f"Step {step}: PSNR={np.mean(losses['psnr']):.2f}, SSIM={np.mean(losses['ssim']):.4f}, LPIPS={np.mean(losses['lpips']):.4f}")

    def render_path(self, step: int):
        if self.cfg.path_type == "none":
            return

        # Use intrinsics and size from the first validation image
        data = self.valset[0]
        K = data["K"].to(self.device)
        image = data["image"].to(self.device)
        width, height = image.shape[1], image.shape[0]  # Width, Height in the right order

        # Fix: Use camtoworlds instead of poses
        # Get poses and bounds from parser
        poses = self.parser.camtoworlds  # (N, 4, 4) numpy array
        # Convert to numpy if it's a tensor
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()

        bounds = np.array([self.cfg.near_plane, self.cfg.far_plane])  # Approximation

        # Generate trajectory based on path_type
        if self.cfg.path_type == "spiral":
            render_poses = generate_spiral_path(
                poses=poses,
                bounds=bounds,
                n_frames=self.cfg.n_frames,
                n_rots=self.cfg.n_rots,
                zrate=self.cfg.zrate,
                spiral_scale_f=self.cfg.spiral_scale_f,
                spiral_scale_r=self.cfg.spiral_scale_r,
                focus_distance=self.cfg.focus_distance,
            )
        elif self.cfg.path_type == "ellipse_z":
            render_poses = generate_ellipse_path_z(
                poses=poses,
                n_frames=self.cfg.n_frames,
                variation=self.cfg.variation,
                phase=self.cfg.phase,
                height=self.cfg.height,
            )
        elif self.cfg.path_type == "ellipse_y":
            render_poses = generate_ellipse_path_y(
                poses=poses,
                n_frames=self.cfg.n_frames,
                variation=self.cfg.variation,
                phase=self.cfg.phase,
                height=self.cfg.height,
            )
        elif self.cfg.path_type == "interpolated":
            render_poses = generate_interpolated_path(
                poses=poses,
                n_interp=self.cfg.n_interp,
                spline_degree=self.cfg.spline_degree,
                smoothness=self.cfg.smoothness,
                rot_weight=self.cfg.rot_weight,
            )
        else:
            raise ValueError(f"Unknown path_type: {self.cfg.path_type}")

        # Get validation poses and images for metric computation
        max_samples = min(len(self.valset), self.cfg.num_val) if self.cfg.num_val > 0 else len(self.valset)
        val_camtoworlds = [self.valset[i]["camtoworld"].to(self.device) for i in range(max_samples)]

        # Get ground truth images in the correct format
        val_images = []
        for i in range(max_samples):
            img = self.valset[i]["image"].to(self.device)
            # Convert to [C, H, W] format
            img = img.permute(2, 0, 1)
            val_images.append(img)

        val_translations = torch.stack([camtoworld[:3, 3] for camtoworld in val_camtoworlds], dim=0)  # (N_val, 3)

        # Initialize lists to store metrics and GT markers
        psnrs = []
        ssims = []
        lpips_list = []
        is_gt_list = []
        threshold = 1e-5  # Threshold to determine if a pose is a GT pose

        # Create video writer
        video_path = f"{self.render_dir}/trajectory_{self.cfg.path_type}.mp4"

        # Create frames directory for temporary storage
        frames_dir = f"{self.render_dir}/frames_{self.cfg.path_type}"
        os.makedirs(frames_dir, exist_ok=True)

        # Render each pose, save images, and calculate metrics
        for i, pose in enumerate(render_poses):
            print(f"Rendering frame {i + 1}/{len(render_poses)}")

            # Ensure the pose is a well-formed 4x4 matrix
            if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
                print(f"WARNING: Invalid pose shape: {pose.shape if hasattr(pose, 'shape') else 'unknown'}")
                # Try to create a valid pose
                if isinstance(pose, np.ndarray) and pose.size >= 12:
                    pose = pose.reshape(3, 4)
                    full_pose = np.eye(4)
                    full_pose[:3, :4] = pose
                    pose = full_pose
                else:
                    print("ERROR: Cannot create a valid pose matrix")
                    continue

            # Convert to tensor and ensure it's on the right device
            camtoworld = torch.from_numpy(pose.astype(np.float32)).to(self.device).unsqueeze(0)
            rendered_trans = camtoworld[0, :3, 3]

            # Find the closest validation pose
            dists = torch.norm(val_translations - rendered_trans, dim=1)
            min_dist, min_idx = torch.min(dists, dim=0)
            is_gt = min_dist < threshold  # Mark as GT if distance is below threshold
            gt_image = val_images[min_idx.item()].unsqueeze(0)  # [1, 3, H, W]

            # Render the image
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworld,
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
            )
            colors = torch.clamp(colors, 0.0, 1.0)  # [1, H, W, 3]

            # 1. Save the rendered image to disk
            if len(colors.shape) == 4:  # [B, H, W, C]
                canvas = colors[0].cpu().numpy()  # Use indexing instead of squeeze
            else:
                print(f"WARNING: Unexpected colors shape: {colors.shape}")
                continue

            canvas = (canvas * 255).astype(np.uint8)

            # Save both to the main directory and the frames directory
            frame_path = f"{frames_dir}/frame_{i:04d}.png"
            imageio.imwrite(frame_path, canvas)
            imageio.imwrite(f"{self.render_dir}/path_{self.cfg.path_type}_{i:04d}.png", canvas)

            # 2. Calculate PSNR, SSIM, and LPIPS - convert colors to match gt_rgb format
            colors_for_metrics = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]

            # Ensure dimensions match
            if colors_for_metrics.shape[2:] != gt_image.shape[2:]:
                import torch.nn.functional as F

                # Check if the dimensions are swapped
                if colors_for_metrics.shape[2] == gt_image.shape[3] and colors_for_metrics.shape[3] == gt_image.shape[
                    2]:
                    # GT has swapped dimensions, transpose it
                    gt_image = gt_image.transpose(2, 3)
                else:
                    # Different dimensions, resize
                    gt_image = F.interpolate(gt_image, size=colors_for_metrics.shape[2:], mode='bilinear',
                                             align_corners=False)

            # Calculate metrics
            psnr = self.psnr_fn(colors_for_metrics, gt_image).item()
            ssim = self.ssim_fn(colors_for_metrics, gt_image).item()
            lpips = self.lpips_fn(colors_for_metrics, gt_image).item()

            # Store metrics and GT status
            psnrs.append(psnr)
            ssims.append(ssim)
            lpips_list.append(lpips)
            is_gt_list.append(is_gt.item())

        # Create video from saved frames
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-framerate', '30',
                '-i', f"{frames_dir}/frame_%04d.png",
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                video_path
            ]
            subprocess.run(cmd)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Failed to create video: {e}")
            print("Try creating video manually with:")
            print(f"ffmpeg -framerate 30 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}")

        # 3. Create plots for each metric with GT poses marked
        gt_indices = [i for i, gt in enumerate(is_gt_list) if gt]

        # PSNR Plot
        plt.figure()
        plt.plot(range(len(psnrs)), psnrs, label='PSNR')
        if gt_indices:  # Only scatter if there are GT poses
            plt.scatter(gt_indices, [psnrs[i] for i in gt_indices], color='red', label='GT Poses')
        plt.xlabel('Frame')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig(f"{self.render_dir}/psnr_{self.cfg.path_type}.png")
        plt.close()

        # SSIM Plot
        plt.figure()
        plt.plot(range(len(ssims)), ssims, label='SSIM')
        if gt_indices:
            plt.scatter(gt_indices, [ssims[i] for i in gt_indices], color='red', label='GT Poses')
        plt.xlabel('Frame')
        plt.ylabel('SSIM')
        plt.legend()
        plt.savefig(f"{self.render_dir}/ssim_{self.cfg.path_type}.png")
        plt.close()

        # LPIPS Plot
        plt.figure()
        plt.plot(range(len(lpips_list)), lpips_list, label='LPIPS')
        if gt_indices:
            plt.scatter(gt_indices, [lpips_list[i] for i in gt_indices], color='red', label='GT Poses')
        plt.xlabel('Frame')
        plt.ylabel('LPIPS')
        plt.legend()
        plt.savefig(f"{self.render_dir}/lpips_{self.cfg.path_type}.png")
        plt.close()

        print(f"Rendered {len(render_poses)} frames for {self.cfg.path_type} trajectory at step {step}")
        print(f"Saved plots to {self.render_dir}")


def main(cfg: Config):
    set_random_seed(42)
    runner = Runner(cfg)
    for step in range(0, cfg.num_steps + 1, cfg.eval_step):
        runner.eval(step)
        if cfg.path_type != "none":
            runner.render_path(step)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)