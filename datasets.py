import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"}


def get_img_names(path: str) -> List[str]:
    """
    Optimized image file retrieval function using pathlib and extension filtering.

    Args:
        path: Directory path to search for images

    Returns:
        List of image file paths as strings
    """
    img_names = []
    path_obj = Path(path)

    for ext in VALID_EXTENSIONS:
        img_names.extend([str(p) for p in path_obj.rglob(f"*{ext}")])
        img_names.extend([str(p) for p in path_obj.rglob(f"*{ext.upper()}")])

    return img_names


class CustomDataset(Dataset):
    """
    Optimized dataset class with lazy loading of images.

    This dataset supports optional preloading for small datasets and uses
    lazy loading for larger datasets to manage memory efficiently.

    Args:
        image_folder: Path to folder containing images
        transform: Optional torchvision transforms to apply
        preload: Whether to preload images into memory (only for small datasets)
    """

    def __init__(self, image_folder: str, transform=None, preload: bool = False):
        self.image_folder = image_folder
        self.image_paths = get_img_names(image_folder)
        self.transform = transform or transforms.Compose(
            [transforms.Resize((299, 299)), transforms.ToTensor()]
        )

        # Only preload for small datasets based on available memory
        self.preload = preload and len(self.image_paths) < 1000

        if self.preload:
            with ThreadPoolExecutor(max_workers=4) as executor:
                self.images = list(executor.map(self._load_image, self.image_paths))
        else:
            self.images = None

    def _load_image(self, img_path: str) -> Image.Image:
        """
        Load a single image file.

        Args:
            img_path: Path to image file

        Returns:
            PIL Image in RGB format
        """
        # Ensure img_path is a string
        if isinstance(img_path, (list, tuple)):
            raise ValueError(f"Expected string path, got {type(img_path)}: {img_path}")

        return Image.open(str(img_path)).convert("RGB")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.image_paths)))
            images = []
            for i in indices:
                if self.preload:
                    image = self.images[i]
                else:
                    image = self._load_image(self.image_paths[i])

                if self.transform:
                    image = self.transform(image)
                images.append(image)
            return images
        else:
            if self.preload:
                image = self.images[idx]
            else:
                image = self._load_image(self.image_paths[idx])

            if self.transform:
                image = self.transform(image)
            return image


class SSIMDataset(Dataset):
    """
    Optimized dataset class with support for GPU preloading.

    This dataset is specifically designed for SSIM calculations and supports
    loading tensors directly to GPU when available.

    Args:
        image_folder: Path to folder containing images
        transform: Optional torchvision transforms to apply
        preload: Whether to preload images into memory (only for small datasets)
        device: Device to load tensors to ('cpu' or 'cuda')
    """

    def __init__(
        self,
        image_folder: str,
        transform=None,
        preload: bool = False,
        device: str = "cpu",
    ):
        self.image_folder = image_folder
        self.image_paths = get_img_names(image_folder)
        self.device = torch.device(device)
        self.transform = transform or transforms.Compose(
            [transforms.Resize((299, 299)), transforms.ToTensor()]
        )

        # Only preload for small datasets based on available memory
        self.preload = preload and len(self.image_paths) < 1000

        if self.preload:
            with ThreadPoolExecutor(max_workers=4) as executor:
                images = list(executor.map(self._load_image, self.image_paths))
                self.images = [img.to(self.device) for img in images]
        else:
            self.images = None

    def _load_image(self, img_path: str) -> torch.Tensor:
        """
        Load a single image file and convert to tensor.

        Args:
            img_path: Path to image file

        Returns:
            Image tensor
        """
        if isinstance(img_path, (list, tuple)):
            raise ValueError(f"Expected string path, got {type(img_path)}: {img_path}")

        image = Image.open(str(img_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.image_paths)))
            images = []
            for i in indices:
                if self.preload:
                    image = self.images[i]
                else:
                    image = self._load_image(self.image_paths[i])
                    if not self.preload and self.device.type == "cuda":
                        image = image.to(self.device)
                images.append(image)
            return images
        else:
            if self.preload:
                image = self.images[idx]
            else:
                image = self._load_image(self.image_paths[idx])
                if not self.preload and self.device.type == "cuda":
                    image = image.to(self.device)
            return image