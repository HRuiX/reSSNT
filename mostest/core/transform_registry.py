"""
Transform Registry and Parameter Configuration
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import albumentations as A
import re


@dataclass
class ParamConfig:
    """Parameter Configuration Class"""
    name: str  # Parameter name
    range: Tuple[float, float]  # Parameter range
    param_type: str  # Parameter type: 'int', 'float', 'symmetric', 'odd_int', 'gamma', 'tuple', 'choice'
    choices: List[Any] = None  # For choice type
    fixed_mode: bool = False  # Fixed mode, eliminates range randomness (for tuple type)
    tuple_format: str = 'auto'  # Tuple format: 'auto', 'range', 'bbox', 'none'
                                 # 'auto': Auto-detect (2 elements as range, 4 elements judged by name)
                                 # 'range': Range parameter (min, max), ensures ascending order
                                 # 'bbox': Bounding box (x1, y1, x2, y2), ensures x1<=x2, y1<=y2
                                 # 'none': No correction

    def _parse_tuple_info(self) -> Tuple[int, List[str]]:
        """
        Parse tuple type information
        Returns: (number of elements, list of element types)
        Example: 'tuple[float, float]' -> (2, ['float', 'float'])
                 'tuple[int, int]' -> (2, ['int', 'int'])
        """
        match = re.search(r'tuple\[(.*?)\]', self.param_type)
        if match:
            types_str = match.group(1)
            types = [t.strip() for t in types_str.split(',')]
            return len(types), types
        return 2, ['float', 'float']  # Default value

    def get_param_count(self) -> int:
        """
        Get the number of normalized values needed for this parameter
        For tuple types, fixed_mode=True requires only 1 value, otherwise returns tuple length
        For other types, returns 1
        """
        if self.param_type.startswith('tuple'):
            if self.fixed_mode:
                return 1  # Fixed mode: only needs one value
            count, _ = self._parse_tuple_info()
            return count
        return 1

    def decode(self, normalized_value) -> Any:
        """
        Decode normalized value [0,1] to actual parameter value
        For tuple types, normalized_value should be an array/list
        For other types, normalized_value is a single float value
        """
        if self.param_type.startswith('tuple'):
            return self._decode_tuple(normalized_value)

        # Single value type processing
        min_val, max_val = self.range
        value = min_val + normalized_value * (max_val - min_val)

        if self.param_type == 'int':
            return int(round(value))
        elif self.param_type == 'odd_int':
            int_val = int(round(value))
            return int_val if int_val % 2 == 1 else int_val + 1
        elif self.param_type == 'symmetric':
            abs_val = abs(value)
            return (-abs_val, abs_val)
        elif self.param_type == 'gamma':
            return (value, value + 20)
        elif self.param_type == 'choice':
            if self.choices:
                idx = int(normalized_value * (len(self.choices) - 1))
                return self.choices[idx]
        else:  # 'float'
            return float(value)

    def _decode_tuple(self, normalized_values) -> Tuple:
        """
        Decode tuple type parameter
        normalized_values: Normalized value array, length should equal tuple element count

        Fixed mode (fixed_mode=True):
            - Uses only 1 normalized value
            - Returns (value, value), eliminates Albumentations' random sampling

        Range mode (fixed_mode=False):
            - Uses multiple normalized values
            - Returns (min, max), Albumentations will randomly sample within this range

        Special handling:
        - If range min is negative, split [0,1] into two parts:
            [0, 0.5) maps to [min_val, 0)
            [0.5, 1] maps to [0, max_val]
        - Correct order based on tuple_format:
            'range': Ensure ascending order (min, max)
            'bbox': Ensure (x1, y1, x2, y2) with x1<=x2, y1<=y2
            'auto': Auto-detect (2 elements as range, 4 elements based on name)
            'none': No correction
        """
        count, elem_types = self._parse_tuple_info()
        min_val, max_val = self.range

        # Ensure normalized_values is iterable
        if not isinstance(normalized_values, (list, tuple, np.ndarray)):
            normalized_values = [normalized_values]

        # Fixed mode: fill entire tuple with the same value
        if self.fixed_mode:
            norm_val = normalized_values[0] if len(normalized_values) > 0 else 0.5
            value = self._map_normalized_to_range(norm_val, min_val, max_val)

            elem_type = elem_types[0] if len(elem_types) > 0 else 'float'
            if 'int' in elem_type:
                value = int(round(value))
            else:
                value = float(value)

            # Return same value, ensuring no randomness
            return tuple([value] * count)

        # Range mode: normal decoding
        result = []
        for i in range(count):
            if i >= len(normalized_values):
                # If not enough values provided, use middle value
                norm_val = 0.5
            else:
                norm_val = normalized_values[i]

            value = self._map_normalized_to_range(norm_val, min_val, max_val)

            # Convert based on element type
            elem_type = elem_types[i] if i < len(elem_types) else 'float'
            if 'int' in elem_type:
                result.append(int(round(value)))
            else:  # float
                result.append(float(value))

        # Correct order based on tuple_format
        result = self._fix_tuple_order(result, count)

        return tuple(result)

    def _fix_tuple_order(self, result: List, count: int) -> List:
        """
        Correct tuple element order based on tuple_format

        Args:
            result: List of decoded values
            count: Number of elements

        Returns:
            Corrected value list
        """
        if len(result) != count:
            return result

        format_type = self.tuple_format

        # Auto mode: auto-detect
        if format_type == 'auto':
            if count == 2:
                # 2 elements default to range
                format_type = 'range'
            elif count == 4:
                # 4 elements: judge based on parameter name
                name_lower = self.name.lower()
                if any(keyword in name_lower for keyword in ['roi', 'bbox', 'box']):
                    format_type = 'bbox'
                else:
                    # Other 4-element tuples not corrected
                    format_type = 'none'
            else:
                format_type = 'none'

        # Apply correction
        if format_type == 'range' and count == 2:
            # Range parameter: ensure ascending order
            if result[0] > result[1]:
                result[0], result[1] = result[1], result[0]

        elif format_type == 'bbox' and count == 4:
            # bbox parameter: ensure x1<=x2, y1<=y2
            x1, y1, x2, y2 = result
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            result = [x1, y1, x2, y2]

        # 'none': No correction, return directly

        return result

    def _map_normalized_to_range(self, norm_val: float, min_val: float, max_val: float) -> float:
        """Map normalized value to actual range"""
        if min_val < 0:
            # Handle range containing negative numbers
            if norm_val < 0.5:
                return min_val + norm_val * 2 * (0 - min_val)
            else:
                return 0 + (norm_val - 0.5) * 2 * max_val
        else:
            return min_val + norm_val * (max_val - min_val)

@dataclass
class TransformConfig:
    """Transform Configuration Class"""
    name: str  # Transform name
    params: List[ParamConfig]  # Parameter list
    semantic_preservation: str  # Semantic preservation: 'high', 'medium', 'low', 'complete'
    fixed_params: Dict[str, Any] = None  # Fixed parameters

    @property
    def num_params(self) -> int:
        """
        Number of parameters
        For tuple type parameters, calculates element count
        """
        total = 0
        for param in self.params:
            total += param.get_param_count()
        return total

    def decode_params(self, normalized_params: np.ndarray) -> Dict[str, Any]:
        """
        Decode parameters
        Decode parameters from normalized values

        normalized_params: Normalized parameter array
        For parameters containing tuple types, consumes multiple normalized values in order
        """
        params_dict = {}

        # Add fixed parameters
        if self.fixed_params:
            # Ensure fixed_params is dict type
            if not isinstance(self.fixed_params, dict):
                raise TypeError(
                    f"fixed_params must be a dict, got {type(self.fixed_params).__name__}. "
                    f"Transform: {self.name}, fixed_params: {self.fixed_params}"
                )
            params_dict.update(self.fixed_params)

        # Decode variable parameters
        idx = 0
        for param_config in self.params:
            param_count = param_config.get_param_count()

            if idx + param_count <= len(normalized_params):
                if param_count == 1:
                    # Single value parameter
                    params_dict[param_config.name] = param_config.decode(normalized_params[idx])
                else:
                    # Tuple parameter, extract multiple values
                    values = normalized_params[idx:idx+param_count]
                    params_dict[param_config.name] = param_config.decode(values)
                idx += param_count
            else:
                # Not enough parameters, skip
                break

        return params_dict

    def create_transform(self, params_dict: Dict[str, Any]) -> A.BasicTransform:
        """
        Create Albumentations transform object
        """
        transform_class = getattr(A, self.name)
        return transform_class(**params_dict, p=1.0)


# Core Transform Configurations
TRANSFORM_CONFIGS = {
    # ============ Geometric Transforms ============
    0: TransformConfig(
        name="SafeRotate",
        params=[
            ParamConfig("limit", (0.001, 90), "float"),
            ],
        semantic_preservation="complete"
    ),

    1: TransformConfig(
        name="Affine",
        params=[
            ParamConfig("scale", (0.5, 2), "float"),
            ParamConfig("rotate", (-45, 45), "float"), 
            ParamConfig("shear", (-15, 15), "float"), 
            ParamConfig("translate_percent", (-0.05, 0.05), "float"),  
        ],
        semantic_preservation="high"
    ),

    2: TransformConfig(
        name="ShiftScaleRotate",
        params=[
            ParamConfig("shift_limit", (0.001, 0.0625), "float"),
            ParamConfig("scale_limit", (0.001, 0.1), "float"),
            ParamConfig("rotate_limit", (1, 45), "int"),
        ],
        semantic_preservation="high"
    ),


    3: TransformConfig(
        name="Rotate",
        params=[ParamConfig("limit", (0.001, 90), "float")],
        semantic_preservation="complete"
    ),

    4: TransformConfig(
        name="HorizontalFlip",
        params=[],
        semantic_preservation="complete"
    ),

    5: TransformConfig(
        name="VerticalFlip",
        params=[],
        semantic_preservation="complete"
    ),

    6: TransformConfig(
        name="Perspective",
        params=[
            ParamConfig("scale", (0.001, 0.15), "float"),
        ],
        semantic_preservation="high"
    ),

    7: TransformConfig(
        name="CoarseDropout",
        params=[
            ParamConfig("num_holes_range", (1, 10), "tuple[int, int]"),
        ],
        fixed_params={
                "hole_height_range": (0.001, 0.3),
                "hole_width_range": (0.001, 0.3),
            }, 
        semantic_preservation="medium"
    ),

    8: TransformConfig(
        name="GridDropout",
        params=[
            ParamConfig("ratio", (0.001, 0.35), "float"),
        ],
        semantic_preservation="medium"
    ),

    9: TransformConfig(
        name="PixelDropout",
        params=[
            ParamConfig("dropout_prob", (0.001, 0.25), "float"),
        ],
        semantic_preservation="medium"
    ),

    10: TransformConfig(
        name="FrequencyMasking",
        params=[
            ParamConfig("freq_mask_param", (1, 200), "int"),
        ],
        semantic_preservation="medium"
    ),

    11: TransformConfig(
        name="TimeMasking",
        params=[ParamConfig("time_mask_param", (1,200), "int")],
        semantic_preservation="medium"
    ),

    12: TransformConfig(
        name="Morphological",
        params=[
            ParamConfig("scale", (10, 15), "int"),
        ],
        semantic_preservation="low"
    ),

    13: TransformConfig(
        name="CLAHE",
        params=[
            ParamConfig("clip_limit", (1, 4), "float"),
        ],
        semantic_preservation="high"
    ),

    14: TransformConfig(
        name="RandomBrightnessContrast",
        params=[
            ParamConfig("brightness_limit", (0.001, 0.25), "float"),
            ParamConfig("contrast_limit", (0.001, 0.25), "float"),
        ],
        semantic_preservation="high"
    ),

    15: TransformConfig(
        name="HueSaturationValue",
        params=[
            ParamConfig("hue_shift_limit", (0.001, 30), "float"),
            ParamConfig("sat_shift_limit", (0.001, 30), "float"),
            ParamConfig("val_shift_limit", (0.001, 30), "float"),
        ],
        semantic_preservation="high"
    ),

    16: TransformConfig(
        name="ColorJitter",
        params=[
            ParamConfig("brightness", (0.001, 1.2), "float"),
            ParamConfig("contrast", (0.001, 1.2), "float"),
            ParamConfig("saturation", (0.001, 1.2), "float"),
            ParamConfig("hue", (0.001, 0.5), "float"),
            ],
        semantic_preservation="high"
    ),

    17: TransformConfig(
        name="RandomGamma",
        params=[ParamConfig("gamma_limit", (70,130), "tuple[int, int]")],
        semantic_preservation="high"
    ),

    18: TransformConfig(
        name="Posterize",
        params=[ParamConfig("num_bits", (5, 7), "int")],
        semantic_preservation="medium"
    ),

    19: TransformConfig(
        name="Solarize",
        params=[ParamConfig("threshold_range", (0.3,0.7), "tuple[float, float]")],
        semantic_preservation="medium"
    ),

    20: TransformConfig(
        name="ChannelDropout",
        params=[],
        fixed_params={"channel_drop_range": [1, 1]},
        semantic_preservation="low"
    ),

    21: TransformConfig(
        name="RandomSunFlare",
        params=[
            ParamConfig("flare_roi", (0.001, 1), "tuple[float, float, float, float]"),
            ParamConfig("angle_range", (0.001, 1), "tuple[float, float]"),
            ],
        fixed_params={"src_radius": 200},
        semantic_preservation="low"
    ),

    22: TransformConfig(
        name="GaussNoise",
        params=[
            ParamConfig("std_range", (0.001, 0.25), "tuple[float, float]"),
            ParamConfig("noise_scale_factor", (0.001, 1), "float"),
        ],
        semantic_preservation="high"
    ),

    23: TransformConfig(
        name="ISONoise",
        params=[
            ParamConfig("color_shift", (0.001, 0.5), "tuple[float, float]"),
            ParamConfig("intensity", (0.001, 0.5), "tuple[float, float]"),
        ],
        semantic_preservation="high"
    ),

    24: TransformConfig(
        name="MultiplicativeNoise",
        params=[
            ParamConfig("multiplier", (0.001, 1.5), "tuple[float, float]"),
        ],
        semantic_preservation="high"
    ),

    25: TransformConfig(
        name="ShotNoise",
        params=[
            ParamConfig("scale_range", (0.001, 0.3), "tuple[float, float]"),
        ],
        semantic_preservation="high"
    ),

    26: TransformConfig(
        name="SaltAndPepper",
        params=[
            ParamConfig("amount", (0.001, 0.06), "tuple[float, float]"),
            ParamConfig("salt_vs_pepper", (0.001, 0.6), "tuple[float, float]"),
        ],
        semantic_preservation="high"
    ),

    27: TransformConfig(
        name="Blur",
        params=[
            ParamConfig("blur_limit", (3, 10), "int"),
        ],
        semantic_preservation="medium"
    ),

    28: TransformConfig(
        name="GaussianBlur",
        params=[
            ParamConfig("sigma_limit", (0.001, 3), "float"),
        ],
        semantic_preservation="medium"
    ),

    29: TransformConfig(
        name="MedianBlur",
        params=[
            ParamConfig("blur_limit", (3, 15), "int"),
        ],
        semantic_preservation="medium"
    ),

    30: TransformConfig(
        name="MotionBlur",
        params=[
            ParamConfig("blur_limit", (3, 20), "int"),
        ],
        semantic_preservation="medium"
    ),

    31: TransformConfig(
        name="AdvancedBlur",
        params=[
            ParamConfig("blur_limit", (3, 15), "int"),
            ParamConfig("rotate_limit", (1, 180), "int"),
            ParamConfig("noise_limit", (0.5, 1.5), "float"),
        ],
        semantic_preservation="medium"
    ),

    32: TransformConfig(
        name="ZoomBlur",
        params=[
            ParamConfig("max_factor", (1, 1.5), "float"),
            ParamConfig("step_factor", (0.01, 0.03), "float"),
        ],
        semantic_preservation="medium"
    ),

    33: TransformConfig(
        name="Defocus",
        params=[
            ParamConfig("radius", (1, 5), "int"),
            ParamConfig("alias_blur", (0.001, 0.25), "float"),
        ],
        semantic_preservation="medium"
    ),

    34: TransformConfig(
        name="GlassBlur",
        params=[
            ParamConfig("sigma", (0.001, 0.65), "float"),
            ParamConfig("max_delta", (1, 3), "int"),
        ],
        fixed_params={"iterations": 1},
        semantic_preservation="low"
    ),
    
    35: TransformConfig(
        name="OpticalDistortion",
        params=[
            ParamConfig("distort_limit", (-0.5, 0.5), "float"),
        ],
        semantic_preservation="medium"
    ),

    36: TransformConfig(
        name="CropAndPad",
        params=[
            ParamConfig("percent", (-0.25, 0.25), "tuple[float, float, float, float]"),
        ],
        semantic_preservation="high"
    ),
    
    37: TransformConfig(
        name="RandomFog",
        params=[
            ParamConfig("alpha_coef", (0.001, 0.1), "float"),
            ParamConfig("fog_coef_range", (0.001, 0.5), "tuple[float, float]"),
        ],
        semantic_preservation="medium"
    ),

    38: TransformConfig(
        name="RandomRain",
        params=[
            ParamConfig("slant_range", (-15, 15), "tuple[float, float]"),
            ParamConfig("drop_length", (1, 50), "int"),
            ParamConfig("brightness_coefficient", (0.001, 1), "float"),
        ],
        semantic_preservation="low"
    ),

    39: TransformConfig(
        name="RandomSnow",
        params=[
            ParamConfig("brightness_coeff", (0.001, 2.5), "float"),
            ParamConfig("snow_point_range", (0.001, 0.3), "tuple[float, float]"),
        ],
        semantic_preservation="low"
    ),

    40: TransformConfig(
        name="Illumination",
        params=[
            ParamConfig("intensity_range", (0.01, 0.2), "tuple[float, float]"),
            ParamConfig("angle_range", (0.001, 360), "tuple[float, float]"),
            ParamConfig("center_range", (0.001, 0.9), "tuple[float, float]"),
            ParamConfig("sigma_range", (0.2, 1), "tuple[float, float]"),
        ],
        semantic_preservation="low"
    ),

    41: TransformConfig(
            name="RandomGravel",
            params=[
                ParamConfig("gravel_roi", (0.001, 1), "tuple[float, float, float, float]"),
                ParamConfig("number_of_patches", (1, 20), "int"),
            ],
            semantic_preservation="low"
        ),

     42: TransformConfig(
                name="RGBShift",
                params=[
                    ParamConfig("r_shift_limit", (1, 20), "int"),
                    ParamConfig("g_shift_limit", (1, 20), "int"),
                    ParamConfig("b_shift_limit", (1, 20), "int"),
                ],
                semantic_preservation="low"
            ),

}


class TransformRegistry:
    """
    Transform Registry Manager
    """

    @staticmethod
    def get_config(transform_id: int) -> TransformConfig:
        """Get transform configuration"""
        if transform_id not in TRANSFORM_CONFIGS:
            raise ValueError(f"Unknown transform ID: {transform_id}")
        return TRANSFORM_CONFIGS[transform_id]

    @staticmethod
    def get_max_params() -> int:
        """Get maximum number of parameters"""
        return max(config.num_params for config in TRANSFORM_CONFIGS.values())

    @staticmethod
    def list_transforms() -> Dict[int, str]:
        """List all transforms"""
        return {tid: config.name for tid, config in TRANSFORM_CONFIGS.items()}

    @staticmethod
    def list_by_category() -> Dict[str, List[int]]:
        """List transforms by category"""
        return {
            "geometric": [0, 1, 2, 3, 4, 5, 41, 42, 43],
            "masking": [6, 7, 8, 9, 10, 11, 12, 13],
            "color": [14, 15, 16, 17, 18, 19, 20],
            "noise": [23, 24, 25, 26, 27],
            "blur": [28, 29, 30, 31, 32, 33, 34, 35],
            "weather": [38, 39, 40],
            "lighting": [22],
            "channel": [21],
            "distortion": [36],
            "crop": [37]
        }

    @staticmethod
    def get_all_configs() -> List[TransformConfig]:
        """
        Get all transform configurations as a list

        Returns:
            List of TransformConfig objects (ordered by transform ID)
        """
        return [TRANSFORM_CONFIGS[tid] for tid in sorted(TRANSFORM_CONFIGS.keys())]


    @staticmethod
    def print_transform_info(transform_id: int):
        """Print detailed transform information for debugging"""
        config = TransformRegistry.get_config(transform_id)
        print(f"\n{'='*60}")
        print(f"Transform ID: {transform_id}")
        print(f"Name: {config.name}")
        print(f"Total normalized params needed: {config.num_params}")
        print(f"Semantic preservation: {config.semantic_preservation}")
        print(f"\nParameters:")
        idx = 0
        for param in config.params:
            count = param.get_param_count()
            if count == 1:
                print(f"  [{idx}] {param.name}: {param.param_type}, range={param.range}")
            else:
                print(f"  [{idx}:{idx+count}] {param.name}: {param.param_type}, range={param.range}")
            idx += count
        if config.fixed_params:
            print(f"\nFixed parameters: {config.fixed_params}")
        print(f"{'='*60}\n")

class SpatialMaskedTransform(A.BasicTransform):
    """
    Spatial Localized Transform Wrapper

    Only applies the transform within a specified rectangular region, keeping other areas unchanged
    """

    def __init__(
        self,
        transform: A.BasicTransform,
        bbox: Tuple[int, int, int, int] = None,  # (x1, y1, x2, y2)
        bbox_normalized: Tuple[float, float, float, float] = None,  # 0-1 range
        always_apply: bool = False,
        p: float = 1.0
    ):
        """
        Args:
            transform: Base transform to apply
            bbox: Absolute coordinate bounding box (x1, y1, x2, y2)
            bbox_normalized: Normalized coordinates (x1, y1, x2, y2), range [0,1]

        Example:
            # Only apply blur to center region of image
            transform = SpatialMaskedTransform(
                A.GaussianBlur(sigma_limit=(5, 5)),
                bbox_normalized=(0.25, 0.25, 0.75, 0.75)  # Center 50% region
            )
        """
        super().__init__(always_apply, p)
        self.transform = transform
        self.bbox = bbox
        self.bbox_normalized = bbox_normalized

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply spatial localized transform"""
        h, w = img.shape[:2]

        if self.bbox_normalized is not None:
            x1 = int(self.bbox_normalized[0] * w)
            y1 = int(self.bbox_normalized[1] * h)
            x2 = int(self.bbox_normalized[2] * w)
            y2 = int(self.bbox_normalized[3] * h)
        elif self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
        else:
            x1, y1, x2, y2 = 0, 0, w, h

        region = img[y1:y2, x1:x2].copy()
        transformed_region = self.transform(image=region)['image']
        result = img.copy()
        result[y1:y2, x1:x2] = transformed_region

        return result

    @property
    def targets(self):
        """Define supported target types"""
        return {"image": self.apply}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("transform", "bbox", "bbox_normalized")


@dataclass
class SpatialParamConfig(ParamConfig):
    """
    Spatial Parameter Configuration
    Extends ParamConfig, adds position information
    """
    spatial_bbox: bool = False  # Whether it's a spatial localization parameter

    def get_param_count(self) -> int:
        """bbox requires 4 normalized values (x1, y1, x2, y2)"""
        if self.spatial_bbox:
            return 4
        return super().get_param_count()

    def decode(self, normalized_value) -> Any:
        """Decode bbox coordinates"""
        if self.spatial_bbox:
            # Ensure bbox is valid: x1 < x2, y1 < y2
            values = list(normalized_value)
            x1, y1, x2, y2 = sorted([values[0], values[2]]), sorted([values[1], values[3]])
            return (x1[0], y1[0], x1[1], y1[1])
        return super().decode(normalized_value)

def get_standard_transforms():
    """
    Get standard transform configuration list (convenience function)

    Returns:
        List of all available TransformConfig objects
    """
    return TransformRegistry.get_all_configs()