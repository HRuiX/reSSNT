import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import random
import sys
from mostest.core.transform_registry import TransformRegistry, TRANSFORM_CONFIGS


def comprehensive_psnr(img1, img2, method='opencv'):
    """
    Comprehensive PSNR calculation function

    Parameters:
    - img1, img2: Input images
    - method: Calculation method ('opencv', 'manual', 'skimage')
    """

    # Preprocessing: ensure image sizes match
    if img1.shape != img2.shape:
        h, w = img2.shape[0], img2.shape[1]
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

    if method == 'opencv':
        try:
            return cv2.PSNR(img1, img2)
        except:
            print("OpenCV calculation failed, using manual method")
            method = 'manual'

    if method == 'manual':
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')

        max_val = 255.0 if img1.max() > 1 else 1.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr


def comp_plot(img1_rgb, img2_rgb):
    # Create subplots for display
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    axes[0].imshow(img1_rgb)
    axes[0].set_title('Image 1')
    axes[0].axis('off')

    axes[1].imshow(img2_rgb)
    axes[1].set_title('Image 2')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show(block=False)


class ImageTransformsOperate(ImageOnlyTransform):
    def __init__(self, operation='', hyper=None, p=1):
        super().__init__(p=p)
        if hyper is not None:
            self.init_variable(operation, hyper)
        else:
            self.init_variable(operation)

    def init_variable(self, operation, hyper=None):
        raise NotImplementedError

    def get_support_types(self):
        raise NotImplementedError

    def get_operation_type(self):
        return [self.operate, self.hyper, self.get_support_types()]

    def set_hyper(self, hyper):
        raise NotImplementedError


class PixelMutation(ImageTransformsOperate):
    """Custom pixel mutation transformation"""

    def init_variable(self, mutation_type, pixel_prob=0.2):
        self.hyper = pixel_prob

        if mutation_type == '':
            mutation_type = random.choice(['single', 'rows', 'cols'])

        self.operate = mutation_type  # 'single', 'rows_cols', or 'both'

    def apply(self, image, **params):
        # Get random pixel positions (coordinates) for 20% of pixels
        height, width, channels = image.shape
        total_pixels = height * width
        modified_image = image.copy()

        if self.operate == 'single':
            # Calculate number of pixels to modify
            num_pixels_to_modify = int(total_pixels * self.hyper)

            # Randomly select pixel positions
            pixel_indices = np.random.choice(total_pixels, num_pixels_to_modify, replace=False)

            # Convert 1D indices to 2D coordinates
            rows = pixel_indices // width
            cols = pixel_indices % width

            # Create mask to mark modified pixels
            mask = np.zeros((height, width), dtype=bool)
            mask[rows, cols] = True

            # Random colors
            modified_image[rows, cols] = np.random.randint(0, 256, size=(num_pixels_to_modify, channels))


        elif self.operate == 'rows':
            num_to_modify = int(height * self.hyper)  # 20% of rows

            # Randomly select row indices to modify
            rows_to_modify = np.random.choice(height, size=num_to_modify, replace=False)

            # Modify selected rows (set to red)
            for row in rows_to_modify:
                modified_image[row, :, :] = [1, 0, 0]  # Red (R=1, G=0, B=0)

        elif self.operate == 'cols':
            # Modify columns
            num_to_modify = int(width * self.hyper)  # 20% of columns

            # Randomly select column indices to modify
            cols_to_modify = np.random.choice(width, size=num_to_modify, replace=False)

            # Modify selected columns (set to blue)
            for col in cols_to_modify:
                modified_image[:, col, :] = [1, 0, 0]  # Blue (R=0, G=0, B=1)

        return modified_image.astype(image.dtype)

    def get_support_types(self):
        return ['single', 'rows', 'cols']

    def set_hyper(self, pixel_prob):
        self.hyper = pixel_prob


class SpectralOperation(ImageTransformsOperate):
    """Custom spectral/channel operation transformation"""

    def init_variable(self, operation_type, noise_scale=20):

        self.hyper = noise_scale

        if operation_type == '':
            self.operate = random.choice(['disappear', 'noise'])

        self.operate = operation_type  # 'single', 'rows_cols', or 'both'

    def apply(self, image, **params):
        num_channels = image.shape[2]
        self.channel = random.randint(0, num_channels - 1)
        modified_image = image.copy()

        if self.operate == 'disappear':
            # Channel disappears
            modified_image[:, :, self.channel] = 0
        else:
            # Channel noise
            noise = np.random.normal(0, self.hyper, modified_image[:, :, self.channel].shape)
            modified_image[:, :, self.channel] = np.clip(modified_image[:, :, self.channel] + noise, 0, 255)

        return modified_image.astype(image.dtype)

    def get_support_types(self):
        return ['disappear', 'noise']

    def set_hyper(self, noise_scale):
        self.hyper = noise_scale


class ImageTransforms(object):
    def __init__(self):
        self.g = False
        self.m = False
        self.create_augmentation_pipeline()
        self.old_idx = []

    # Define various augmentation transformations
    def create_augmentation_pipeline(self):
        # Get all transformation configs from TransformRegistry

        # Geometric transforms (IDs: 0-5)
        self.geometric_transforms = self._create_transforms_from_ids([0, 1, 2, 3, 4, 5, 6])

        # Masking and occlusion (IDs: 7-12)
        self.masking_ops = self._create_transforms_from_ids([7, 8, 9, 10, 11, 12])

        # Other operations (IDs: 13-42)
        other_ids = list(range(13, 43))
        self.other_ops = self._create_transforms_from_ids(other_ids)

    def _create_transforms_from_ids(self, transform_ids):
        """
        Create Albumentations transform objects from list of transform IDs
        Uses random parameters to maintain diversity

        Args:
            transform_ids: List of transform IDs

        Returns:
            List of Albumentations transform objects
        """

        transforms = []
        for tid in transform_ids:
            try:
                config = TransformRegistry.get_config(tid)

                # Generate random normalized parameters [0, 1]
                num_params = config.num_params
                normalized_params = np.random.rand(num_params)

                # Decode parameters
                params_dict = config.decode_params(normalized_params)

                # Create transform object
                transform = config.create_transform(params_dict)
                transforms.append(transform)

            except Exception as e:
                print(f"Warning: Failed to create transform {tid}: {e}")
                continue

        return transforms

    def _refresh_transform(self, transform_id: int):
        """
        Refresh transform with specified ID by generating new random parameters

        Args:
            transform_id: Transform ID

        Returns:
            New transform object, or None if failed
        """
        try:
            config = TransformRegistry.get_config(transform_id)

            # Generate new random parameters
            num_params = config.num_params
            normalized_params = np.random.rand(num_params)

            # Decode and create transform
            params_dict = config.decode_params(normalized_params)
            transform = config.create_transform(params_dict)

            return transform

        except Exception as e:
            print(f"Warning: Failed to refresh transform {transform_id}: {e}")
            return None

    def __call__(self, image, TRY_NUM, mask=None, *args, **kwargs):
        """
        Apply image transformation - maintains original interface

        Args:
            image: Input image
            TRY_NUM: Number of attempts
            mask: Optional mask

        Returns:
            (transformed_image, mask, transform_name) or None (if all attempts fail)
        """
        # Build operations dict
        all_ops = {"Other": self.other_ops}
        if not self.g:
            all_ops.update({"geometric_transforms": self.geometric_transforms})
        if not self.m:
            all_ops.update({"masking_ops": self.masking_ops})

        # Track attempted operations to avoid duplicates
        tried_operations = set()

        for i in range(1, TRY_NUM + 1):
            try:
                # Get all category keys
                key = list(all_ops.keys())

                # Build all possible operation combinations (category_idx, op_idx, category_name)
                all_possible_ops = []
                for cat_idx, category in enumerate(key):
                    ops = all_ops[category]
                    for op_idx in range(len(ops)):
                        if category == "Other":
                            # Deduplication check for Other category
                            if op_idx not in self.old_idx:
                                all_possible_ops.append((cat_idx, op_idx, category))
                        else:
                            all_possible_ops.append((cat_idx, op_idx, category))

                # If all Other operations are used, reset and add again
                if not any(op for op in all_possible_ops if op[2] == "Other") and "Other" in key:
                    self.old_idx = []
                    for op_idx in range(len(all_ops["Other"])):
                        cat_idx = key.index("Other")
                        all_possible_ops.append((cat_idx, op_idx, "Other"))

                available_ops = [op for op in all_possible_ops if op not in tried_operations]

                if not available_ops:
                    continue

                ops_idx, op_idx, selected_key = random.choice(available_ops)
                tried_operations.add((ops_idx, op_idx, selected_key))

                if selected_key == "geometric_transforms":
                    self.g = True
                elif selected_key == "masking_ops":
                    self.m = True

                ops = all_ops[selected_key]
                op = ops[op_idx]

                # Execute transformation
                if mask is not None:
                    transforms = A.Compose([op])
                    res = transforms(image=image, mask=mask)
                else:
                    res = op(image=image)
                    if "mask" not in res:
                        res["mask"] = None

                transform_name = str(op).split("(")[0]

                if selected_key == "Other":
                    self.old_idx.append(op_idx)

                # Check if image was actually modified
                if np.array_equal(res["image"], image):
                    continue

                return res["image"], res["mask"], transform_name

            except Exception as e:
                continue

        return None

    def get_transform_by_name(self, name: str, params= None):
        """
        Get transform object by name (optional feature for external precise control)

        Args:
            name: Transform name
            params: Optional parameter dict

        Returns:
            Transform object or None
        """
        # Search for matching transform in configs
        for tid, config in TRANSFORM_CONFIGS.items():
            if config.name == name:
                if params is None:
                    # Use random parameters
                    num_params = config.num_params
                    normalized_params = np.random.rand(num_params)
                    params = config.decode_params(normalized_params)

                return config.create_transform(params)

        return None

    def reset_state(self):
        """Reset state flags (optional convenience method)"""
        self.g = False
        self.m = False
        self.old_idx = []

    def get_available_transforms(self):
        """
        Get statistics of currently available transforms

        Returns:
            Number of transforms in each category
        """
        return {
            "geometric_transforms": len(self.geometric_transforms) if not self.g else 0,
            "masking_ops": len(self.masking_ops) if not self.m else 0,
            "other_ops": len(self.other_ops) - len(self.old_idx)
        }