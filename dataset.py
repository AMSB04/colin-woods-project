import os
import random
from typing import Optional, Tuple, List, Dict
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from model_util import set_seed, get_device

class MRISliceDataset(Dataset):
    """
    Custom Dataset for loading 2D axial MRI slices from 3T and 7T folders.

    Supports both paired and unpaired configurations. Applies optional augmentations
    during training and normalizes intensity values to a specified range.

    Args:
        root_dir (str): Root directory containing '3T/<mode>' and '7T/<mode>' folders.
        mode (str): One of ['train', 'val', 'test'] to determine data split.
        paired (bool): Whether to enforce that 3T and 7T images are paired by filename.
        target_size (Tuple[int, int]): Target size (H, W) to resize slices to.
        norm_range (Tuple[float, float]): Desired intensity normalization range.
        augment (bool): Whether to apply data augmentation (only if mode == 'train').
        return_metadata (bool): If True, includes filenames and flags in the output.
        valid_exts (Tuple[str]): Allowed image file extensions.
    """
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        paired: bool = False,
        target_size: Tuple[int, int] = (256, 256),
        norm_range: Tuple[float, float] = (-1, 1),
        augment: bool = True,
        return_metadata: bool = False,
        valid_exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp')
    ):
        self.paired = paired
        self.augment = augment and mode == 'train'
        self.norm_range = norm_range
        self.return_metadata = return_metadata
        self.mode = mode
        self.target_size = target_size
        self.device = get_device()

        # Validate and set up directories
        self.dir_3t = os.path.join(root_dir, '3T', mode)
        self.dir_7t = os.path.join(root_dir, '7T', mode)

        if not os.path.isdir(self.dir_3t) or not os.path.isdir(self.dir_7t):
            raise ValueError(f"Missing required folders in {root_dir}: {self.dir_3t} or {self.dir_7t}")

        # Get sorted file lists
        self.files_3t = sorted([f for f in os.listdir(self.dir_3t) if f.lower().endswith(valid_exts)])
        self.files_7t = sorted([f for f in os.listdir(self.dir_7t) if f.lower().endswith(valid_exts)])

        if not self.files_3t or not self.files_7t:
            raise ValueError(f"No valid images found in {self.dir_3t} or {self.dir_7t}")

        # Identify common filenames for paired setup
        if paired:
            self.common_files = sorted(list(set(self.files_3t) & set(self.files_7t)))
            if not self.common_files:
                raise ValueError("No matching files found for paired dataset")

        # Base transforms (always applied - resizing, tensor conversion, and normalization)
        self.base_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) if norm_range == (-1, 1) else transforms.Lambda(lambda x: x)
        ])

        # Initialize random augmentations
        self.random_transform = transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ], p=0.8)  # 80% chance to apply at least one augmentation

    def __len__(self):
        if self.paired:
            return len(self.common_files)
        return max(len(self.files_3t), len(self.files_7t))

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Loads and transforms an image from the given path.

        Args:
            path (str): File path to the image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        try:
            with Image.open(path) as img:
                img = img.convert('L')
                tensor = self.base_transform(img)
                if self.augment:
                    # Apply different random seed for each load
                    set_seed(random.randint(0, 2**32-1))
                    tensor = self.random_transform(tensor)
                return tensor
            
        except (IOError, OSError, UnidentifiedImageError) as e:
            raise UnidentifiedImageError(f"Cannot identify image file {path}")

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns a sample consisting of a 3T and 7T image (and optional metadata).

        Args:
            idx (int): Index of the data point.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'A' = 3T image, 'B' = 7T image.
        """
        try:
            if self.paired:
                fname = self.common_files[idx]
                path_3t = os.path.join(self.dir_3t, fname)
                path_7t = os.path.join(self.dir_7t, fname)
            else:
                fname_3t = self.files_3t[idx % len(self.files_3t)]
                fname_7t = self.files_7t[idx % len(self.files_7t)]
                path_3t = os.path.join(self.dir_3t, fname_3t)
                path_7t = os.path.join(self.dir_7t, fname_7t)

            image_3t = self._load_image(path_3t)
            image_7t = self._load_image(path_7t)

            item = {'A': image_3t, 'B': image_7t}

            if self.return_metadata:
                item['metadata'] = {
                    'filename_3T': os.path.basename(path_3t),
                    'filename_7T': os.path.basename(path_7t),
                    'paired': self.paired,
                    'mode': self.mode
                }

            return item
        except UnidentifiedImageError as e:
            raise UnidentifiedImageError(str(e))
        except Exception as e:
            raise RuntimeError(f"Error accessing sample {idx}: {str(e)}")

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to combine a batch of samples.

        Args:
            batch (List[Dict]): List of samples from __getitem__.

        Returns:
            Dict[str, torch.Tensor]: Batched data dictionary.
        """
        out = {
            'A': torch.stack([item['A'] for item in batch]),
            'B': torch.stack([item['B'] for item in batch])
        }
        if 'metadata' in batch[0]:
            out['metadata'] = [item['metadata'] for item in batch]
        return out