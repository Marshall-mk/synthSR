"""
Data loading and generation for SynthSR training with MONAI.

This module provides high-level data generation pipelines that convert high-resolution
MRI scans into paired low-resolution/high-resolution training data using domain
randomization techniques.
"""

import torch
from typing import Optional, List

from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandSpatialCropd,
    SpatialPadd,
    ToTensord,
    GaussianSmooth,
    ScaleIntensityRangePercentiles,
    CenterSpatialCropd,
)

from .domain_rand import (
    SampleResolution,
    MimicAcquisition,
    RandomSpatialDeformation,
    BiasFieldCorruption,
    IntensityAugmentation,
)


class HRLRDataGenerator:
    """
    Domain randomization pipeline for generating paired LR-HR training data.

    Converts high-resolution MRI images to paired low-resolution/high-resolution data
    following the SynthSR approach. Applies configurable augmentations including spatial
    deformations, bias field corruption, intensity augmentation, and resolution randomization
    to create diverse and realistic training pairs.

    **Training Philosophy:**
    - HR (target): Minimally augmented to preserve ground truth quality
    - LR (input): Heavily augmented with realistic degradations to simulate diverse acquisition protocols

    **Pipeline Steps:**
    1. Start with HR image
    2. Create LR path: Deformation → Bias field → Intensity aug → Resolution degradation
    3. Create HR path: Keep original or apply minimal augmentations
    4. Normalize both to [0, 1] range
    5. Return (LR, HR) pair

    Args:
        atlas_res: Resolution of input HR images in mm [x, y, z].
                  Example: [1.0, 1.0, 1.0] for 1mm isotropic
        target_res: Target output resolution in mm [x, y, z].
                   Usually same as atlas_res. Default: [1.0, 1.0, 1.0]
        output_shape: Output spatial shape [D, H, W].
                     Example: [128, 128, 128]
        min_resolution: Minimum (highest quality) resolution for randomization [x, y, z].
                       Default: [1.0, 1.0, 1.0]
        max_res_aniso: Maximum anisotropic resolution [x, y, z].
                      Higher = lower quality. Default: [9.0, 9.0, 9.0]
        randomise_res: If True, randomize acquisition resolution.
                      If False, use fixed atlas_res. Default: True
        apply_lr_deformation: If True, apply spatial deformations to LR.
                             Default: True
        apply_bias_field: If True, apply bias field corruption to LR.
                         Default: True
        apply_intensity_aug: If True, apply intensity augmentation to LR.
                            Default: True
        enable_90_rotations: If True, enable 90° rotations in spatial deformation.
                            Default: False
        clip_to_unit_range: If True, clip outputs to [0, 1] range.
                           Default: True

    Example:
        >>> generator = HRLRDataGenerator(
        ...     atlas_res=[1.0, 1.0, 1.0],
        ...     output_shape=[128, 128, 128],
        ...     randomise_res=True,
        ...     apply_lr_deformation=True
        ... )
        >>> hr_batch = torch.randn(2, 1, 128, 128, 128)  # Simulated HR data
        >>> lr_batch, hr_aug_batch = generator.generate_paired_data(hr_batch)
        >>> print(lr_batch.shape, hr_aug_batch.shape)  # Both (2, 1, 128, 128, 128)
        >>> # lr_batch has degradations, hr_aug_batch is clean target
    """

    def __init__(
        self,
        atlas_res=[1.0, 1.0, 1.0],
        target_res=[1.0, 1.0, 1.0],
        output_shape=[128, 128, 128],
        min_resolution=[1.0, 1.0, 1.0],
        max_res_aniso=[9.0, 9.0, 9.0],
        randomise_res=True,
        apply_lr_deformation=True,
        apply_bias_field=True,
        apply_intensity_aug=True,
        enable_90_rotations=False,
        clip_to_unit_range=True,
    ):
        self.atlas_res = atlas_res
        self.target_res = target_res
        self.output_shape = output_shape
        self.randomise_res = randomise_res
        self.apply_lr_deformation = apply_lr_deformation
        self.apply_bias_field = apply_bias_field
        self.apply_intensity_aug = apply_intensity_aug
        self.enable_90_rotations = enable_90_rotations
        self.clip_to_unit_range = clip_to_unit_range

        # Augmentation components
        if apply_lr_deformation:
            self.deformer = RandomSpatialDeformation(
                scaling_bounds=0.15,
                rotation_bounds=15,
                shearing_bounds=0.012,
                translation_bounds=10,
                enable_90_rotations=enable_90_rotations,
                elastic_sigma_range=(5.0, 7.0),
                elastic_magnitude_range=(100.0, 200.0),
                prob_deform=1.0,
            )

        if apply_bias_field:
            self.bias = BiasFieldCorruption(
                bias_field_std=0.3,
                bias_scale=0.025,
                prob=1.0,
            )

        if apply_intensity_aug:
            self.intensity_aug = IntensityAugmentation(
                clip=300,
                gamma_std=0.5,
                channel_wise=False,
                prob_gamma=0.95,
            )

        self.fixed_blur = GaussianSmooth(sigma=0.5)

        if randomise_res:
            self.res_sampler = SampleResolution(
                min_resolution=min_resolution,
                max_res_iso=[1.0, 1.0, 1.0],
                max_res_aniso=max_res_aniso,
                prob_iso=0.02,
                prob_min=0.1,
                return_thickness=True,
            )
            self.mimic = MimicAcquisition(
                volume_res=atlas_res,
                target_res=target_res,
                output_shape=output_shape,
                noise_std=0.01,
                prob_noise=1.0,
                build_dist_map=False,
            )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize images to [0, 1] range using percentile scaling.

        Applies per-sample normalization to ensure consistent intensity ranges.
        Uses 0th (min) to 100th (max) percentile for robust normalization.

        Args:
            image: Input tensor of shape (B, C, D, H, W)

        Returns:
            Normalized tensor of shape (B, C, D, H, W) with values in [0, 1]
        """
        batch_size = image.shape[0]
        normalized = []

        for b in range(batch_size):
            img = image[b : b + 1]

            normalizer = ScaleIntensityRangePercentiles(
                lower=0,  # 0th percentile = min
                upper=100,  # 100th percentile = max
                b_min=0.0,  # Output min
                b_max=1.0,  # Output max
                clip=True,
            )
            img_norm = normalizer(img)
            normalized.append(img_norm)

        return torch.cat(normalized, dim=0)

    def generate_paired_data(
        self,
        hr_images: torch.Tensor,
        return_resolution: bool = False,
    ):
        """
        Generate paired LR-HR training data from real HR images.

        Applies domain randomization to create realistic low-resolution inputs
        paired with high-resolution targets. The LR path includes spatial deformations,
        bias field corruption, intensity augmentation, and resolution degradation,
        while the HR path remains clean.

        **Processing Steps:**
        1. Clone HR as both LR and HR paths
        2. LR path: Apply deformation → bias field → intensity aug → resolution degradation
        3. HR path: Keep original (minimal augmentation)
        4. Normalize both to [0, 1]
        5. Optionally clip to [0, 1] for stability

        Args:
            hr_images: High-resolution input images, shape (B, C, D, H, W)
            return_resolution: If True, also returns resolution and thickness tensors.
                              Default: False

        Returns:
            If return_resolution=False:
                (lr_images, hr_augmented): Tuple of tensors, both (B, C, D, H, W)
                - lr_images: Low-resolution degraded inputs in [0, 1]
                - hr_augmented: High-resolution targets in [0, 1]

            If return_resolution=True:
                (lr_images, hr_augmented, resolution, thickness): Tuple of 4 tensors
                - lr_images: Shape (B, C, D, H, W)
                - hr_augmented: Shape (B, C, D, H, W)
                - resolution: Shape (B, 3) - sampled acquisition resolution in mm
                - thickness: Shape (B, 3) - sampled slice thickness in mm

        Example:
            >>> hr_batch = torch.randn(4, 1, 128, 128, 128)
            >>> lr_batch, hr_batch = generator.generate_paired_data(hr_batch)
            >>> print(lr_batch.min(), lr_batch.max())  # Should be in [0, 1]
            >>> # lr_batch has realistic degradations, hr_batch is clean
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # === HIGH-RESOLUTION PATH (Target) ===
        # Keep original HR as clean target
        hr_augmented = hr_images.clone()

        # === LOW-RESOLUTION PATH (Input) ===
        lr_images = hr_augmented.clone()

        # Apply spatial deformations to LR
        if self.apply_lr_deformation:
            lr_images = self.deformer(lr_images, interpolation="bilinear")

        # Apply bias field corruption to LR
        if self.apply_bias_field:
            lr_images = self.bias(lr_images)

        # Apply intensity augmentation to LR
        if self.apply_intensity_aug:
            lr_images = self.intensity_aug(lr_images)
            lr_images = self.fixed_blur(lr_images)

        # Normalize both HR and LR to [0, 1]
        hr_augmented = self._normalize_image(hr_augmented)
        lr_images = self._normalize_image(lr_images)

        # Domain randomization: simulate different acquisition protocols
        resolution = None
        thickness = None

        if self.randomise_res:
            # Sample random acquisition parameters
            resolution, thickness = self.res_sampler(batch_size)
            resolution = resolution.to(device)
            thickness = thickness.to(device)

            # Simulate acquisition (blur, downsample, noise, upsample)
            lr_images = self.mimic(lr_images, resolution, thickness)
        else:
            # Use fixed atlas resolution
            resolution = torch.tensor(
                [self.atlas_res] * batch_size, dtype=torch.float32, device=device
            )
            thickness = resolution.clone()
            
        lr_images = self._normalize_image(lr_images)
        # Final clipping to ensure [0, 1] range (training stability)
        if self.clip_to_unit_range:
            lr_images = torch.clamp(lr_images, 0.0, 1.0)
            hr_augmented = torch.clamp(hr_augmented, 0.0, 1.0)

        if return_resolution:
            return lr_images, hr_augmented, resolution, thickness
        else:
            return lr_images, hr_augmented

    def generate_lr_only(
        self,
        hr_images: torch.Tensor,
        return_resolution: bool = False,
    ):
        """
        Generate low-resolution images only (for inference/testing).

        Applies the same LR degradation pipeline as generate_paired_data, but
        without generating the HR target. Useful for testing the degradation
        pipeline or generating synthetic LR data for evaluation.

        Args:
            hr_images: High-resolution input images, shape (B, C, D, H, W)
            return_resolution: If True, also returns resolution and thickness tensors.
                              Default: False

        Returns:
            If return_resolution=False:
                lr_images: Low-resolution images, shape (B, C, D, H, W)

            If return_resolution=True:
                (lr_images, resolution, thickness): Tuple of 3 tensors
                - lr_images: Shape (B, C, D, H, W)
                - resolution: Shape (B, 3)
                - thickness: Shape (B, 3)

        Example:
            >>> hr_batch = torch.randn(4, 1, 128, 128, 128)
            >>> lr_batch = generator.generate_lr_only(hr_batch)
            >>> # lr_batch contains only degraded LR images
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        lr_images = hr_images.clone()

        # Apply deformation
        if self.apply_lr_deformation:
            lr_images = self.deformer(lr_images, interpolation="bilinear")

        # Apply bias field
        if self.apply_bias_field:
            lr_images = self.bias(lr_images)

        # Apply intensity augmentation
        if self.apply_intensity_aug:
            lr_images = self.intensity_aug(lr_images)
            lr_images = self.fixed_blur(lr_images)

        # Normalize LR images
        lr_images = self._normalize_image(lr_images)

        # Domain randomization
        if self.randomise_res:
            resolution, thickness = self.res_sampler(batch_size)
            resolution = resolution.to(device)
            thickness = thickness.to(device)
            lr_images = self.mimic(lr_images, resolution, thickness)
        else:
            resolution = torch.tensor(
                [self.atlas_res] * batch_size, dtype=torch.float32, device=device
            )
            thickness = resolution.clone()

        # Final clipping
        if self.clip_to_unit_range:
            lr_images = torch.clamp(lr_images, 0.0, 1.0)

        if return_resolution:
            return lr_images, resolution, thickness
        else:
            return lr_images


class GeneratorDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that generates LR-HR pairs on-the-fly.

    Takes preprocessed HR images from the base MONAI dataset and applies
    HRLRDataGenerator to create paired LR-HR training data.
    """

    def __init__(self, base_dataset, generator, return_resolution):
        self.base_dataset = base_dataset
        self.generator = generator
        self.return_resolution = return_resolution

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Get LR-HR pair for training.

        Returns:
            If return_resolution=False:
                (lr_image, hr_image): Tuple of tensors, each (C, D, H, W)
            If return_resolution=True:
                (lr_image, hr_image, resolution, thickness)
        """
        data = self.base_dataset[idx]
        hr_image = data["image"]

        # Ensure proper shape (B, C, D, H, W) for generator
        if hr_image.ndim == 3:
            hr_image = hr_image.unsqueeze(0)
        hr_image = hr_image.unsqueeze(0)  # Add batch dim

        # Generate paired data
        result = self.generator.generate_paired_data(
            hr_image,
            return_resolution=self.return_resolution,
        )

        # Unpack and remove batch dimension
        if self.return_resolution:
            lr_image, hr_augmented, resolution, thickness = result
            lr_image = lr_image.squeeze(0)
            hr_augmented = hr_augmented.squeeze(0)
            resolution = resolution.squeeze(0)
            thickness = thickness.squeeze(0)
            return lr_image, hr_augmented, resolution, thickness
        else:
            lr_image, hr_augmented = result
            lr_image = lr_image.squeeze(0)
            hr_augmented = hr_augmented.squeeze(0)
            return lr_image, hr_augmented


def create_dataset(
    image_paths: List[str],
    generator: HRLRDataGenerator,
    target_shape: Optional[List[int]] = None,
    target_spacing: Optional[List[float]] = None,
    use_cache: bool = False,
    return_resolution: bool = False,
    is_training: bool = True,
):
    """
    Create MONAI Dataset for HR-LR data generation with preprocessing pipeline.

    Builds a complete data loading and preprocessing pipeline using MONAI transforms,
    then wraps it with HRLRDataGenerator for on-the-fly LR-HR pair generation during training.

    **Preprocessing Pipeline:**
    1. LoadImaged: Load NIfTI files
    2. EnsureChannelFirstd: Ensure (C, D, H, W) format
    3. Orientationd: Align to RAS orientation (consistent coordinate system)
    4. Spacingd: Resample to target resolution (if specified)
    5. CropForegroundd: Remove background
    6. RandSpatialCropd/SpatialPadd: Match target shape
    7. ToTensord: Convert to PyTorch tensors

    Then, HRLRDataGenerator wraps this to produce (LR, HR) pairs on-the-fly.

    Args:
        image_paths: List of paths to HR NIfTI images
        generator: HRLRDataGenerator instance for LR-HR pair generation
        target_shape: Target spatial shape [D, H, W].
                     If None, no shape matching is applied.
                     Example: [128, 128, 128]
        target_spacing: Target voxel spacing in mm [x, y, z].
                       If None, no resampling is applied.
                       Example: [1.0, 1.0, 1.0]
        use_cache: If True, use CacheDataset to cache preprocessed images in memory.
                  Faster training but requires more RAM. Default: False
        return_resolution: If True, dataset returns (lr, hr, resolution, thickness).
                          If False, returns (lr, hr). Default: False
        is_training: If True, use random spatial cropping for augmentation.
                    If False, use center padding. Default: True

    Returns:
        GeneratorDataset: PyTorch Dataset that returns:
            - If return_resolution=False: (lr_image, hr_image)
            - If return_resolution=True: (lr_image, hr_image, resolution, thickness)

        Each item shape: lr_image and hr_image are (C, D, H, W)

    Example:
        >>> from pathlib import Path
        >>> image_paths = sorted([str(p) for p in Path("/data/hr").glob("*.nii.gz")])
        >>> generator = HRLRDataGenerator(
        ...     atlas_res=[1.0, 1.0, 1.0],
        ...     output_shape=[128, 128, 128]
        ... )
        >>> dataset = create_dataset(
        ...     image_paths=image_paths,
        ...     generator=generator,
        ...     target_shape=[128, 128, 128],
        ...     target_spacing=[1.0, 1.0, 1.0],
        ...     use_cache=False
        ... )
        >>> print(len(dataset))  # Number of images
        >>> lr, hr = dataset[0]  # Get first LR-HR pair
        >>> print(lr.shape, hr.shape)  # Both (1, 128, 128, 128)
    """
    # Create data dictionary for MONAI
    data_dicts = [{"image": img_path} for img_path in image_paths]

    # Build preprocessing transforms
    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        # Ensure consistent RAS orientation across heterogeneous data
        Orientationd(keys=["image"], axcodes="RAS"),
    ]

    # Resample to target spacing (atlas resolution)
    if target_spacing is not None:
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=target_spacing,
                mode="bilinear",
            )
        )

    # Crop/pad to target shape
    if target_shape is not None:
        transforms.append(CropForegroundd(keys=["image"], source_key="image"))

        if is_training:
            # Training: random crop, then pad if needed
            transforms.append(
                RandSpatialCropd(
                    keys=["image"], roi_size=target_shape, random_size=False
                )
            )
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))
        else:
            # Validation: center crop or pad to exact target shape
            # First try to crop to target shape (if image is larger)
            transforms.append(
                CenterSpatialCropd(keys=["image"], roi_size=target_shape)
            )
            # Then pad if needed (if image is smaller after crop)
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))

    transforms.append(ToTensord(keys=["image"]))

    transform = Compose(transforms)

    # Create base dataset (with optional caching)
    if use_cache:
        dataset = CacheDataset(
            data=data_dicts,
            transform=transform,
            cache_rate=1.0,
            num_workers=4,
        )
    else:
        dataset = Dataset(data=data_dicts, transform=transform)

    # Wrap with generator for LR-HR pair generation 
    return GeneratorDataset(dataset, generator, return_resolution)
