import os
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import imageio.plugins.pyav
from PIL import Image
import multiprocessing as mp
import cv2
import logging
import numpy as np
import torch
from pycolmap import SceneManager

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


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
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
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
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
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
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

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        
        # Only create mask paths if mask directory exists
        image_mask_paths = []
        if os.path.exists(mask_dir):
            image_mask_paths = [os.path.join(mask_dir, colmap_to_image[f].replace('.jpg', '.png')) for f in image_names]
        else:
            print(f"Warning: Mask folder {mask_dir} does not exist. Proceeding without masks.")
            image_mask_paths = [None] * len(image_names)

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
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

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.image_mask_paths = image_mask_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        # scene_scale from "Improving Adaptive Density Control for 3D Gaussian Splatting"
        # https://arxiv.org/pdf/2503.14274
        dists = np.linalg.norm(self.points - scene_center, axis=1)
        self.scene_scale = np.sum(dists, axis=0) / self.points.shape[0]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dataset(torch.utils.data.Dataset):
    """
    A dataset class that:
    - Tracks training iteration through a shared counter for dynamic behavior
    - Returns N nearest cameras for each requested index based on camera positions
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
        # Disable PyAV to avoid potential issues with imageio
        try:
            import imageio.plugins.pyav
            imageio.plugins.pyav.HAVE_AV = False
        except (ImportError, AttributeError):
            try:
                import imageio.v2
                imageio.v2.plugins.pyav.HAVE_AV = False
            except (ImportError, AttributeError):
                try:
                    imageio.plugins.freeimage.download()
                except Exception as e:
                    logger.warning(f"Failed to configure imageio plugins: {e}")

        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_masks = load_masks
        self._current_iter = mp.Value('i', current_iter)  # Shared counter for iteration tracking

        # Determine indices based on split
        indices = np.arange(len(self.parser.image_names))
        if self.parser.test_every <= 0:  # Adjusted to handle test_every == -1 or 0
            self.indices = indices
        else:
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]

        # Precompute camera positions for neighbor selection
        self.all_positions = []
        for idx in self.indices:
            c2w = self.parser.camtoworlds[idx]
            self.all_positions.append(c2w[:3, 3])
        self.all_positions = np.array(self.all_positions)  # Shape: [num_indices, 3]

    def get_downsample_factor(self) -> int:
        """Determine downsample factor based on current iteration."""
        current_iter = self._current_iter.value
        if current_iter < 500:
            return 1
        elif current_iter < 1000:
            return 4
        elif current_iter < 1500:
            return 2
        elif current_iter < 2000:
            return 1
        elif current_iter < 2500:
            return 4
        return 1

    def get_neighboring_indices(self, base_idx: int, N: int) -> List[int]:
        """Return indices of N nearest cameras based on Euclidean distance."""
        base_pos = self.all_positions[base_idx]
        distances = np.linalg.norm(self.all_positions - base_pos, axis=1)
        neighbor_order = np.argsort(distances)
        neighbor_indices = neighbor_order[:N].tolist()
        return neighbor_indices

    def downsample_image(self, image: np.ndarray, factor: int) -> np.ndarray:
        """Downsample an image by a given factor."""
        if factor == 1:
            return image
        h, w = image.shape[:2]
        new_h, new_w = h // factor, w // factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def adjust_intrinsics(self, K: np.ndarray, factor: int) -> np.ndarray:
        """Adjust camera intrinsics for downsampling."""
        if factor == 1:
            return K
        K = K.copy()
        K[0:2, :] /= factor
        return K

    def _try_load_image(self, index: int) -> np.ndarray:
        """Try to load an image with multiple methods."""
        image_path = self.parser.image_paths[index]
        
        # Try imageio first
        try:
            return imageio.imread(image_path)[..., :3]
        except Exception as e:
            logger.warning(f"imageio failed for {image_path}: {e}")
            
        # Try OpenCV next
        try:
            img = cv2.imread(image_path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"OpenCV failed for {image_path}: {e}")
            
        # Try PIL as last resort
        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.warning(f"PIL failed for {image_path}: {e}")
            
        raise RuntimeError(f"All methods failed to load image: {image_path}")

    def _load_mask(self, index: int, image_shape: tuple) -> np.ndarray:
        """Load or create image mask with proper error handling."""
        if self.parser.image_mask_paths[index] is not None:
            try:
                image_mask = imageio.imread(self.parser.image_mask_paths[index])
                if len(image_mask.shape) == 3:
                    if image_mask.shape[2] == 4:  # RGBA
                        image_mask = image_mask[..., -1]  # Take alpha channel
                    else:  # RGB
                        image_mask = image_mask.mean(axis=2)
                return (image_mask > 127).astype(np.uint8)
            except Exception as e:
                logger.warning(f"Failed to load mask {self.parser.image_mask_paths[index]}: {e}")
        
        # Fallback to dummy mask
        return np.ones(image_shape[:2], dtype=np.uint8)

    def get_data(self, item: int, patch_coords: Optional[tuple] = None) -> tuple:
        """Retrieve data for a single index, optionally using provided patch coordinates."""
        idx = self.indices[item]
        downsample_factor = self.get_downsample_factor()

        # Load image and mask
        image = self._try_load_image(idx)
        image_mask = self._load_mask(idx, image.shape)
        camera_id = self.parser.camera_ids[idx]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[idx]
        mask = self.parser.mask_dict.get(camera_id)  # Use get() for safety

        # Undistort if necessary
        if len(params) > 0 and self.parser.mapx_dict[camera_id] is not None:
            try:
                mapx, mapy = self.parser.mapx_dict[camera_id], self.parser.mapy_dict[camera_id]
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y:y + h, x:x + w]
                image_mask = image_mask[y:y + h, x:x + w]
            except Exception as e:
                logger.warning(f"Undistortion failed for index {idx}: {e}")

        # Apply downsampling
        if downsample_factor > 1:
            image = self.downsample_image(image, downsample_factor)
            K = self.adjust_intrinsics(K, downsample_factor)
            if image_mask is not None:
                image_mask = self.downsample_image(image_mask.astype(np.float32), downsample_factor) > 0.5

        # Handle patch cropping
        if self.patch_size is not None:
            ds_patch_size = self.patch_size // downsample_factor
            h, w = image.shape[:2]
            if patch_coords is None:
                rand_x = np.random.randint(0, max(w - ds_patch_size, 1))
                rand_y = np.random.randint(0, max(h - ds_patch_size, 1))
            else:
                rand_x, rand_y = patch_coords
                # Clamp coordinates to ensure valid cropping
                rand_x = min(max(rand_x, 0), w - ds_patch_size)
                rand_y = min(max(rand_y, 0), h - ds_patch_size)

            image = image[rand_y:rand_y + ds_patch_size, rand_x:rand_x + ds_patch_size]
            if image_mask is not None:
                image_mask = image_mask[rand_y:rand_y + ds_patch_size, rand_x:rand_x + ds_patch_size]
            K_adjusted = K.copy()
            K_adjusted[0, 2] -= rand_x
            K_adjusted[1, 2] -= rand_y
        else:
            rand_x, rand_y = 0, 0
            K_adjusted = K

        # Prepare base data
        data = {
            "K": torch.from_numpy(K_adjusted).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_mask": ~torch.from_numpy(image_mask).bool() if image_mask is not None else None,
            "image_id": item,
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Load mask from file if specified
        if self.load_masks:
            mask_path = self.parser.image_paths[idx].replace("images", "masks").rsplit(".", 1)[0] + ".jpg"
            try:
                mask_from_file = imageio.imread(mask_path)
                if len(mask_from_file.shape) == 3:
                    mask_from_file = mask_from_file[..., 0]
                if downsample_factor > 1:
                    mask_from_file = self.downsample_image(mask_from_file.astype(np.float32), downsample_factor) > 0.5
                if self.patch_size is not None:
                    mask_from_file = mask_from_file[rand_y:rand_y + ds_patch_size, rand_x:rand_x + ds_patch_size]
                data["mask"] = torch.from_numpy(mask_from_file).bool()
            except Exception as e:
                logger.warning(f"Failed to load mask from {mask_path}: {e}")

        # Compute depths if specified
        if self.load_depths:
            try:
                worldtocams = np.linalg.inv(camtoworlds)
                image_name = self.parser.image_names[idx]
                point_indices = self.parser.point_indices[image_name]
                points_world = self.parser.points[point_indices]
                points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
                points_proj = (K_adjusted @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]
                depths = points_cam[:, 2]
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
                data["points"] = torch.from_numpy(points).float()
                data["depths"] = torch.from_numpy(depths).float()
            except Exception as e:
                logger.warning(f"Depth computation failed for index {idx}: {e}")
                data["points"] = torch.zeros((0, 2), dtype=torch.float32)
                data["depths"] = torch.zeros(0, dtype=torch.float32)

        return data, (rand_x, rand_y) if self.patch_size is not None else None

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Return data for the item and its N nearest neighbors."""
        # Get data for the first neighbor and establish patch coordinates
        data, patch_coords = self.get_data(item)

        return data

    def update_iteration(self, current_iter: int):
        """Update the shared iteration counter."""
        with self._current_iter.get_lock():
            self._current_iter.value = current_iter

if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
