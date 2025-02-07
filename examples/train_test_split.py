import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import multiprocessing

def process_camera_dir(args):
    """
    Process a single camera directory and split its images according to validation ratio.
    
    Args:
        args (tuple): (root_dir, cam_folder, val_ratio)
    Returns:
        tuple: (train_files, val_files) for this camera
    """
    root_dir, cam_folder, val_ratio = args
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    cam_path = os.path.join(root_dir, cam_folder)
    if os.path.isdir(cam_path):
        for img in os.listdir(cam_path):
            ext = Path(img).suffix.lower()
            if ext in valid_extensions:
                relative_path = os.path.join(cam_folder, img)
                image_files.append(relative_path)
    
    # Shuffle and split for this camera directory
    random.shuffle(image_files)
    val_size = int(len(image_files) * val_ratio)
    
    return (
        image_files[val_size:],  # train files
        image_files[:val_size]   # val files
    )

def create_dataset_files(root_dir, val_ratio=0.2):
    """
    Creates training and validation file lists maintaining val_ratio per camera folder.
    Files are created in the parent directory of root_dir.
    
    Args:
        root_dir (str): Path to the root directory containing cam0, cam1, etc.
        val_ratio (float): Ratio of images to use for validation (0.0 to 1.0)
    """
    # Get parent directory to place output files
    parent_dir = os.path.dirname(os.path.abspath(root_dir))
    output_train = os.path.join(parent_dir, "training_images.txt")
    output_val = os.path.join(parent_dir, "validation_images.txt")
    
    # Get list of camera directories
    cam_dirs = [(root_dir, d, val_ratio) for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))]
    
    # Use ProcessPoolExecutor for parallel processing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Process each camera directory in parallel
        results = list(executor.map(process_camera_dir, cam_dirs))
    
    # Separate train and val files
    train_files = list(chain.from_iterable(train for train, _ in results))
    val_files = list(chain.from_iterable(val for _, val in results))
    
    if not train_files and not val_files:
        raise ValueError(f"No valid images found in {root_dir}")
    
    # Write training file
    with open(output_train, 'w') as f:
        for file_path in train_files:
            f.write(f"{file_path}\n")
    
    # Write validation file
    with open(output_val, 'w') as f:
        for file_path in val_files:
            f.write(f"{file_path}\n")
    
    # Print statistics per camera
    print(f"\nDataset statistics:")
    for (_, cam_folder, _), (train, val) in zip(cam_dirs, results):
        print(f"\n{cam_folder}:")
        print(f"  Training samples: {len(train)}")
        print(f"  Validation samples: {len(val)}")
        if len(train) + len(val) > 0:
            actual_ratio = len(val) / (len(train) + len(val))
            print(f"  Actual validation ratio: {actual_ratio:.3f}")
    
    print(f"\nTotal:")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"\nFiles created in {parent_dir}")
    print(f"Processing completed using {num_cores} cores")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split image dataset into training and validation sets')
    parser.add_argument('--root', type=str, default='images', 
                        help='Root directory containing camera folders')
    parser.add_argument('--val-ratio', type=float, default=0.02, 
                        help='Ratio of images to use for validation (default: 0.02)')
    
    args = parser.parse_args()
    
    create_dataset_files(
        root_dir=args.root,
        val_ratio=args.val_ratio
    )