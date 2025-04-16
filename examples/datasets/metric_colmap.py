import os
import json
import laspy
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import assert_never
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor, as_completed
import open3d as o3d

from PIL import Image
import multiprocessing as mp
import cv2
import logging
import numpy as np
import torch
from pycolmap import SceneManager
from scipy.spatial import cKDTree

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        lidar_data_path: Optional[str] = None,
        total_iterations: int = 30_000,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im.camera_id
            camera_ids.append(camera_id)

            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Distortion
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            else:
                camtype = "unknown"

            mask_dict[camera_id] = None
            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")
        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Sort by filename
        image_names = [imdata[k].name for k in imdata]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        mask_dir = os.path.join(data_dir, "masks" + image_dir_suffix)

        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # Only create mask paths if mask directory exists
        image_mask_paths = []
        if os.path.exists(mask_dir):
            image_mask_paths = [os.path.join(mask_dir, colmap_to_image[f].replace('.jpg', '.png'))
                                for f in image_names]
        else:
            print(f"Warning: Mask folder {mask_dir} does not exist. Proceeding without masks.")
            image_mask_paths = [None] * len(image_names)

        # Load 3D points
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)

        point_indices = {}
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Optionally normalize
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.image_mask_paths = image_mask_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform

        # Make sure there's at least one valid image
        if len(self.image_paths) == 0:
            raise ValueError("No valid images remain after filtering or initialization.")

        # Adjust intrinsics vs actual resolution of the first image
        test_img = cv2.imread(self.image_paths[0], cv2.IMREAD_COLOR)
        if test_img is None:
            raise ValueError("Failed to read any image after filtering for resolution check.")
        actual_height, actual_width = test_img.shape[:2]

        c_id = self.camera_ids[0]
        colmap_width, colmap_height = self.imsize_dict[c_id]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            Ks_dict[camera_id] = K
            w_, h_ = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(w_ * s_width), int(h_ * s_height))

        # Undistortion (if needed)
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask_ = None
            elif camtype == "fisheye":
                fx_ = K[0, 0]
                fy_ = K[1, 1]
                cx_ = K[0, 2]
                cy_ = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx_) / fx_
                y1 = (grid_y - cy_) / fy_
                theta = np.sqrt(x1**2 + y1**2)
                r_ = (1.0 + params[0] * theta**2 + params[1] * theta**4
                      + params[2] * theta**6 + params[3] * theta**8)
                mapx = (fx_ * x1 * r_ + width // 2).astype(np.float32)
                mapy = (fy_ * y1 * r_ + height // 2).astype(np.float32)
                mask_ = (mapx > 0) & (mapy > 0) & (mapx < (width-1)) & (mapy < (height-1))
                ys, xs = np.nonzero(mask_)
                y_min, y_max = ys.min(), ys.max() + 1
                x_min, x_max = xs.min(), xs.max() + 1
                mask_ = mask_[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                # If for some reason the camera type was unknown, skip
                continue

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask_

        # scene scale
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(self.points - scene_center, axis=1)
        self.scene_scale = np.sum(dists, axis=0) / self.points.shape[0]


class Dataset(torch.utils.data.Dataset):
    """
    A dataset class that:
    - Tracks training iteration through a shared counter for dynamic behavior
    - Returns data for an image index (and can find neighbors if needed)
    - Supports robust image loading with multiple fallback methods
    - Handles downsampling, patch cropping, and depth computation
    """

    def __init__(
        self,
        parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        load_masks: bool = False,
        current_iter: int = 0,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_masks = load_masks
        self._current_iter = mp.Value('i', current_iter)


        indices = np.arange(len(self.parser.image_names))
        if self.parser.test_every <= 0:
            self.indices = indices
        else:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]

        # Precompute camera positions
        self.all_positions = []
        for idx in self.indices:
            c2w = self.parser.camtoworlds[idx]
            self.all_positions.append(c2w[:3, 3])
        self.all_positions = np.array(self.all_positions)

    def downsample_image(self, image: np.ndarray, factor: float) -> np.ndarray:
        h, w = image.shape[:2]
        new_h = int(round(h * factor))
        new_w = int(round(w * factor))
        if new_h < 2 or new_w < 2:
            new_h = max(new_h, 2)
            new_w = max(new_w, 2)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _try_load_image(self, index: int) -> np.ndarray:
        """Load an image with multiple fallback methods, but primarily cv2."""
        path = self.parser.image_paths[index]
        # Try OpenCV first
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb

        # Fallback to PIL
        try:
            pil_img = Image.open(path).convert('RGB')
            return np.array(pil_img)
        except Exception as e:
            logger.warning(f"All methods failed to load image: {path}. Error: {e}")
            raise RuntimeError(f"Cannot load image: {path}")

    def _load_mask(self, index: int, image_shape: tuple) -> np.ndarray:
        """Load or create image mask with proper error handling."""
        mask_path = self.parser.image_mask_paths[index]
        if mask_path is not None and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                logger.warning(f"Failed to load mask {mask_path}")
                return np.ones(image_shape[:2], dtype=np.uint8)

            # Convert to a single channel if necessary
            if len(mask_img.shape) == 3:
                if mask_img.shape[2] == 4:  # BGRA
                    mask_img = mask_img[..., -1]
                else:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            return (mask_img > 127).astype(np.uint8)
        # Fallback: full mask
        return np.ones(image_shape[:2], dtype=np.uint8)

    def get_down_factor_for_step(self, step: int) -> float:
        """
        Return the resolution down_factor for the given global 'step'
        by walking through the cumulative schedule.
        """
        return 1.0

    def get_data(self, item: int, patch_coords: Optional[tuple] = None) -> tuple:
        """Retrieve data for a single index, optionally using provided patch coordinates."""
        idx = self.indices[item]
        image = self._try_load_image(idx)
        image_mask = self._load_mask(idx, image.shape)
        camera_id = self.parser.camera_ids[idx]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        c2w = self.parser.camtoworlds[idx]
        mask_ = self.parser.mask_dict.get(camera_id)

        # Possibly undistort
        if len(params) > 0 and camera_id in self.parser.mapx_dict:
            mapx = self.parser.mapx_dict[camera_id]
            mapy = self.parser.mapy_dict[camera_id]
            x, y, w_, h_ = self.parser.roi_undist_dict[camera_id]
            undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            image = undist[y:y + h_, x:x + w_]
            if image_mask is not None:
                undist_mask = cv2.remap(image_mask, mapx, mapy, cv2.INTER_NEAREST)
                image_mask = undist_mask[y:y + h_, x:x + w_]

        # Apply schedule-based downsampling
        downsample_factor = self.get_down_factor_for_step(self._current_iter.value)
        if downsample_factor < 1.0:
            image = self.downsample_image(image, downsample_factor)
            K[0:2, :] *= downsample_factor
            if image_mask is not None:
                image_mask = self.downsample_image(image_mask.astype(np.float32), downsample_factor) > 0.5

        # Handle patch cropping
        if self.patch_size is not None:
            ds_patch_size = int(round(self.patch_size))
            h, w = image.shape[:2]
            if patch_coords is None:
                rand_x = np.random.randint(0, max(w - ds_patch_size, 1))
                rand_y = np.random.randint(0, max(h - ds_patch_size, 1))
            else:
                rand_x, rand_y = patch_coords
                rand_x = min(max(rand_x, 0), w - ds_patch_size)
                rand_y = min(max(rand_y, 0), h - ds_patch_size)

            image = image[rand_y:rand_y + ds_patch_size, rand_x:rand_x + ds_patch_size]
            if image_mask is not None:
                image_mask = image_mask[rand_y:rand_y + ds_patch_size, rand_x:rand_x + ds_patch_size]
            K_adj = K.copy()
            K_adj[0, 2] -= rand_x
            K_adj[1, 2] -= rand_y
        else:
            rand_x, rand_y = 0, 0
            K_adj = K

        data = {
            "K": torch.from_numpy(K_adj).float(),
            "camtoworld": torch.from_numpy(c2w).float(),
            "image": torch.from_numpy(image).float(),
            "image_mask": ~torch.from_numpy(image_mask).bool() if image_mask is not None else None,
            "image_id": item,
            "downsample_factor": downsample_factor,
        }
        if mask_ is not None:
            data["mask"] = torch.from_numpy(mask_).bool()

        if self.load_masks:
            # Attempt to load a separate mask from "masks" folder, if needed
            alt_mask_path = self.parser.image_paths[idx].replace("images", "masks").rsplit(".", 1)[0] + ".jpg"
            if os.path.exists(alt_mask_path):
                alt_mask = cv2.imread(alt_mask_path, cv2.IMREAD_GRAYSCALE)
                if alt_mask is not None:
                    if downsample_factor < 1.0:
                        alt_mask = self.downsample_image(alt_mask.astype(np.float32), downsample_factor) > 0.5
                    if self.patch_size is not None:
                        alt_mask = alt_mask[rand_y:rand_y + ds_patch_size, rand_x:rand_x + ds_patch_size]
                    data["mask"] = torch.from_numpy(alt_mask).bool()

        # Depths
        if self.load_depths:
            try:
                w2c = np.linalg.inv(c2w)
                im_name = self.parser.image_names[idx]
                p_indices = self.parser.point_indices.get(im_name, [])
                p_world = self.parser.points[p_indices]
                p_cam = (w2c[:3, :3] @ p_world.T + w2c[:3, 3:4]).T
                p_proj = (K_adj @ p_cam.T).T
                proj_xy = p_proj[:, :2] / (p_proj[:, 2:3] + 1e-8)
                depths_ = p_cam[:, 2]
                h_, w_ = image.shape[:2]
                sel = (
                    (proj_xy[:, 0] >= 0)
                    & (proj_xy[:, 0] < w_)
                    & (proj_xy[:, 1] >= 0)
                    & (proj_xy[:, 1] < h_)
                    & (depths_ > 0)
                )
                proj_xy = proj_xy[sel]
                depths_ = depths_[sel]
                data["points"] = torch.from_numpy(proj_xy).float()
                data["depths"] = torch.from_numpy(depths_).float()
            except Exception as e:
                logger.warning(f"Depth computation failed for index {idx}: {e}")
                data["points"] = torch.zeros((0, 2), dtype=torch.float32)
                data["depths"] = torch.zeros(0, dtype=torch.float32)

        return data, (rand_x, rand_y) if self.patch_size is not None else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data, _ = self.get_data(item)
        return data

    def update_iteration(self, current_iter: int):
        with self._current_iter.get_lock():
            self._current_iter.value = current_iter

if __name__ == "__main__":
    import argparse

    parser_cli = argparse.ArgumentParser()
    parser_cli.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser_cli.add_argument("--factor", type=int, default=4)
    args = parser_cli.parse_args()

    # Parse COLMAP data
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    if len(dataset) > 0:
        sample_data = dataset[0]
        sample_img = sample_data["image"].numpy().astype(np.uint8)
        h_, w_ = sample_img.shape[:2]
        out_path = "results/points_opencv.mp4"
        os.makedirs("results", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 30, (w_, h_))

        for data in tqdm(dataset, desc="Plotting points"):
            image = data["image"].numpy().astype(np.uint8)
            points = data.get("points", torch.zeros((0, 2))).numpy()
            for x, y in points:
                cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
            bgr_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        writer.release()
        print(f"Video saved at {out_path}")
    else:
        print("No images in dataset. Skipping video creation.")
