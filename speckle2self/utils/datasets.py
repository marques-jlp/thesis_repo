import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import albumentations as A

from utils.image_ops import resize_image, linear_normalization  


class BaseDenoisingDataset(Dataset):
    """
    Base class for denoising datasets with shared utilities.
    """
    def __init__(self, interp='linear'):
        self.interp = interp
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.3),
            # A.ElasticTransform(alpha=1, sigma=50, interpolation=1, border_mode=4, p=0.4)
        ], additional_targets={
            'image0': 'image',
            'image1': 'image',
            'mask': 'image'
        })

    def preprocess_image(self, image):
        """
        Resize and normalize image at different scales.
        """
        image_low = resize_image(image, 0.25, interpol=self.interp)
        image_mid = resize_image(image, 0.5, interpol=self.interp)
        image_high = image

        image_low = linear_normalization(image_low)
        image_mid = linear_normalization(image_mid)
        image_high = linear_normalization(image_high)

        return image_low, image_mid, image_high

    def to_tensor(self, *images):
        """
        Convert numpy images (H, W) to PyTorch tensors (1, H, W).
        """
        return [torch.from_numpy(np.expand_dims(img, 0)) for img in images]


class DenoisingDatasetCCA(BaseDenoisingDataset):
    """
    Dataset for CCA-based denoising using unsupervised low/mid/high resolution inputs.
    Loads a single numpy array of shape (N, H, W).
    """
    def __init__(self, image_dir, interp='linear'):
        super().__init__(interp)
        self.images = np.load(os.path.join(image_dir, "train_data.npy"))  # shape (N, H, W)
        self.num_images = self.images.shape[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_raw = self.images[idx]
        image_low, image_mid, image_high = self.preprocess_image(image_raw)

        transformed = self.transform(
            image=image_low, image0=image_high, image1=image_mid
        )

        image_low, image_high, image_mid = self.to_tensor(
            transformed['image'], transformed['image0'], transformed['image1']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
            'image_mid': image_mid
        }


class DenoisingDatasetSimulator(BaseDenoisingDataset):
    """
    Dataset for simulated denoising with paired noisy and clean images.
    Loads a single numpy array of shape (N, 2, H, W).
    """
    def __init__(self, path, interp='linear'):
        super().__init__(interp)
        data = np.load(os.path.join(path, "train_data.npy"))  # shape (N, 2, H, W)
        self.noisy_imgs = data[:, 0]
        self.clean_imgs = data[:, 1]
        self.num_images = self.noisy_imgs.shape[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        noisy = self.noisy_imgs[idx]
        clean = linear_normalization(self.clean_imgs[idx])

        image_low, image_mid, image_high = self.preprocess_image(noisy)

        transformed = self.transform(
            image=image_low,
            image0=image_high,
            image1=image_mid,
            mask=clean
        )

        image_low, image_high, image_mid, image_clean = self.to_tensor(
            transformed['image'],
            transformed['image0'],
            transformed['image1'],
            transformed['mask']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
            'image_mid': image_mid,
            'image_clean': image_clean
        }


class DenoisingDatasetPNG(BaseDenoisingDataset):
    """
    Dataset for loading PNG images directly from a directory.
    Recursively searches for PNG files in nested patient/cross_sections directories.
    Each PNG file is treated as an individual image for denoising.
    
    Expected structure:
    image_dir/
    ├── patient_1/
    │   └── cross_sections/
    │       ├── image_1.png
    │       └── ...
    ├── patient_2/
    │   └── cross_sections/
    │       └── ...
    └── ...
    """
    def __init__(self, image_dir, interp='linear'):
        super().__init__(interp)
        self.image_dir = image_dir
        self.image_paths = []

        print(f"[DATASET] Initializing PNG dataset", flush=True)
        print(f"[DATASET] Looking in directory: {image_dir}", flush=True)
        
        if not os.path.exists(image_dir):
            raise RuntimeError(f"[DATASET ERROR] Directory does not exist: {image_dir}")

        # Recursively search for PNG files in all subdirectories
        for root, dirs, files in os.walk(image_dir):
            print(f"Scanning: {len(files)} files found", flush=True)

            for file in files:
                if file.lower().endswith('.png'):
                    png_path = os.path.join(root, file)
                    self.image_paths.append(png_path)
        
        # Sort for consistency across runs
        self.image_paths.sort()
        self.num_images = len(self.image_paths)
        
        if self.num_images == 0:
            raise ValueError(f"No PNG files found in {image_dir}")
        
        print(f"Loaded {self.num_images} PNG images from {image_dir}")

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Load image as grayscale
        image_raw = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if image_raw is None:
            raise RuntimeError(f"Failed to load image: {self.image_paths[idx]}")
        
        # Convert to float for processing
        image_raw = image_raw.astype(np.float32)
        
        image_low, image_mid, image_high = self.preprocess_image(image_raw)

        transformed = self.transform(
            image=image_low, image0=image_high, image1=image_mid
        )

        image_low, image_high, image_mid = self.to_tensor(
            transformed['image'], transformed['image0'], transformed['image1']
        )

        return {
            'image_low': image_low,
            'image_high': image_high,
            'image_mid': image_mid
        }