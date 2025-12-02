"""
Data loading and generation for SynthSR training with frequency-domain downsampling.

This module provides an alternative to data.py using FFT-based downsampling
instead of MONAI's spatial-domain resize. The interface is identical to data.py,
allowing easy switching by changing the import in train.py.

Key differences from data.py:
- Uses FFT k-space cropping for downsampling (more realistic MRI simulation)
- Downsamples along a single axis (anisotropic) rather than multi-axis
- Better matches clinical MRI acquisition physics
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
    RandomSpatialDeformation,
    BiasFieldCorruption,
    IntensityAugmentation,
)


def freq_domain_downsample_torch(
    volume: torch.Tensor, downsample_axis: int, factor: float
) -> torch.Tensor:
    """
    Downsample a 3D volume along one axis using frequency domain cropping.

    This simulates k-space undersampling in MRI by:
    1. Performing FFT along all spatial dimensions
    2. Cropping the frequency spectrum (k-space cropping)
    3. Performing inverse FFT

    Args:
        volume: 3D volume tensor of shape (C, D, H, W)
        downsample_axis: Axis to downsample (0=D, 1=H, 2=W)
        factor: Downsampling factor (e.g., 4.0 for 4x downsampling)

    Returns:
        Downsampled volume tensor (size reduced along specified axis)

    Example:
        >>> hr_vol = torch.randn(1, 256, 256, 256)
        >>> lr_vol = freq_domain_downsample_torch(hr_vol, downsample_axis=0, factor=4.0)
        >>> print(lr_vol.shape)  # torch.Size([1, 64, 256, 256])
    """
    # Get original shape
    original_shape = volume.shape

    # Calculate new size along downsampling axis
    # Add 1 to downsample_axis since we have channel dimension first
    spatial_axis = downsample_axis + 1
    new_size = int(round(original_shape[spatial_axis] / factor))

    # Perform FFT along all spatial dimensions
    fft_volume = torch.fft.fftn(volume, dim=(1, 2, 3))
    fft_volume = torch.fft.fftshift(fft_volume, dim=(1, 2, 3))

    # Calculate cropping indices for the frequency domain
    # Crop symmetrically around center (DC component)
    center_idx = original_shape[spatial_axis] // 2
    crop_start = center_idx - new_size // 2
    crop_end = crop_start + new_size

    # Create slice objects for cropping in k-space
    if downsample_axis == 0:  # D axis
        cropped_fft = fft_volume[:, crop_start:crop_end, :, :]
    elif downsample_axis == 1:  # H axis
        cropped_fft = fft_volume[:, :, crop_start:crop_end, :]
    else:  # W axis
        cropped_fft = fft_volume[:, :, :, crop_start:crop_end]

    # Inverse FFT to get downsampled volume
    cropped_fft = torch.fft.ifftshift(cropped_fft, dim=(1, 2, 3))
    downsampled = torch.fft.ifftn(cropped_fft, dim=(1, 2, 3))

    # Take real part (imaginary should be negligible)
    downsampled = torch.real(downsampled)

    return downsampled


class FrequencyDomainDownsample(torch.nn.Module):
    """
    MRI-realistic downsampling using frequency domain (k-space) cropping.

    This simulates actual MRI acquisition by cropping in k-space rather than
    spatial domain interpolation, which better matches MRI physics.

    Args:
        volume_res: Original volume resolution in mm [x, y, z]
        target_res: Target output resolution in mm [x, y, z]
        output_shape: Output spatial shape [D, H, W]
        noise_std: Standard deviation of Gaussian noise to add
        prob_noise: Probability of adding noise

    Example:
        >>> downsample = FrequencyDomainDownsample(
        ...     volume_res=[1.0, 1.0, 1.0],
        ...     target_res=[1.0, 1.0, 1.0],
        ...     output_shape=[128, 128, 128],
        ...     noise_std=0.01
        ... )
        >>> hr_image = torch.randn(1, 1, 128, 128, 128)
        >>> acq_res = torch.tensor([[1.0, 1.0, 4.0]])  # 4x anisotropic in Z
        >>> lr_image = downsample(hr_image, acq_res)
    """

    def __init__(
        self,
        volume_res: List[float],
        target_res: List[float],
        output_shape: List[int],
        noise_std: float = 0.01,
        prob_noise: float = 0.95,
    ):
        super().__init__()
        self.volume_res = torch.tensor(volume_res, dtype=torch.float32)
        self.target_res = torch.tensor(target_res, dtype=torch.float32)
        self.output_shape = output_shape
        self.noise_std = noise_std
        self.prob_noise = prob_noise

    def forward(
        self,
        image: torch.Tensor,
        acquisition_res: torch.Tensor,
        thickness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply frequency-domain downsampling to simulate MRI acquisition.

        Args:
            image: Input high-resolution tensor of shape (B, C, D, H, W)
            acquisition_res: Target acquisition resolution, shape (B, 3) or (3,)
            thickness: Optional slice thickness (not used in this implementation)

        Returns:
            Downsampled tensor of shape (B, C, D, H, W)
        """
        batch_size = image.shape[0]
        device = image.device
        outputs = []

        for b in range(batch_size):
            img = image[b]  # Shape: (C, D, H, W)
            acq_res = (
                acquisition_res[b] if acquisition_res.ndim > 1 else acquisition_res
            )
            acq_res = acq_res.to(device)

            # Determine which axis to downsample (axis with highest anisotropy)
            # Calculate downsampling factors for each axis
            factors = acq_res.to(device) / self.volume_res.to(device)

            # Find axis with maximum downsampling factor (most anisotropic)
            downsample_axis = torch.argmax(factors).item()
            factor = factors[downsample_axis].item()

            # Only downsample if factor > 1.1 (avoid unnecessary processing)
            if factor > 1.1:
                # Apply frequency domain downsampling along the selected axis
                # Don't upsample in k-space - it smooths artifacts
                img = freq_domain_downsample_torch(img, downsample_axis, factor)

                # Upsample back to target shape using NEAREST NEIGHBOR
                # This preserves aliasing artifacts by replicating pixels without smoothing
                if list(img.shape[1:]) != self.output_shape:
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=self.output_shape,
                        mode="nearest", 
                    ).squeeze(0)

            # Add Gaussian noise (simulates scanner noise)
            if self.noise_std > 0 and torch.rand(1).item() < self.prob_noise:
                noise = torch.randn_like(img) * self.noise_std
                img = img + noise

            outputs.append(img.unsqueeze(0))

        return torch.cat(outputs, dim=0)


class HRLRDataGenerator:
    """
    Domain randomization pipeline using frequency-domain downsampling.

    This is an alternative to the spatial-domain approach in data.py.
    Uses FFT-based k-space cropping for more realistic MRI simulation.

    Args:
        atlas_res: Resolution of input HR images in mm [x, y, z]
        target_res: Target output resolution in mm [x, y, z]
        output_shape: Output spatial shape [D, H, W]
        min_resolution: Minimum (highest quality) resolution [x, y, z]
        max_res_aniso: Maximum anisotropic resolution [x, y, z]
        randomise_res: If True, randomize acquisition resolution
        apply_lr_deformation: If True, apply spatial deformations to LR
        apply_bias_field: If True, apply bias field corruption to LR
        apply_intensity_aug: If True, apply intensity augmentation to LR
        enable_90_rotations: If True, enable 90° rotations
        clip_to_unit_range: If True, clip outputs to [0, 1] range
        preserve_background: If True, keep background clean (no FFT artifacts)
        background_threshold: Threshold for brain/background separation (default: 0.01)
    """

    def __init__(
        self,
        atlas_res: list = [1.0, 1.0, 1.0],
        target_res: list = [1.0, 1.0, 1.0],
        output_shape: list = [128, 128, 128],
        min_resolution: list = [1.0, 1.0, 1.0],
        max_res_aniso: list = [9.0, 9.0, 9.0],
        randomise_res: bool = True,
        apply_lr_deformation: bool = True,
        apply_bias_field: bool = True,
        apply_intensity_aug: bool = True,
        enable_90_rotations: bool = False,
        clip_to_unit_range: bool = True,
        preserve_background: bool = True,
        background_threshold: float = 0.073,
    ):
        self.atlas_res = atlas_res
        self.target_res = target_res
        self.output_shape = output_shape
        self.randomise_res = randomise_res
        self.apply_lr_deformation = apply_lr_deformation
        self.apply_bias_field = apply_bias_field
        self.apply_intensity_aug = apply_intensity_aug
        self.clip_to_unit_range = clip_to_unit_range
        self.preserve_background = preserve_background
        self.background_threshold = background_threshold

        # Resolution sampler
        if randomise_res:
            self.res_sampler = SampleResolution(
                min_resolution=min_resolution,
                max_res_iso=None,  # Only use anisotropic
                max_res_aniso=max_res_aniso,
                prob_iso=0.0,  # Force anisotropic for frequency domain method
                prob_min=0.05,
                return_thickness=True,
            )

        # Spatial deformation
        if apply_lr_deformation:
            self.deformer = RandomSpatialDeformation(
                scaling_bounds=0.15,
                rotation_bounds=15.0,
                shearing_bounds=0.012,
                translation_bounds=False,
                enable_90_rotations=enable_90_rotations,
                elastic_sigma_range=(5.0, 7.0),
                elastic_magnitude_range=(100.0, 200.0),
                prob_deform=1.0,
            )

        # Bias field corruption
        if apply_bias_field:
            self.bias = BiasFieldCorruption(
                bias_field_std=0.3, bias_scale=0.025, prob=0.98
            )

        # Intensity augmentation
        if apply_intensity_aug:
            self.intensity_aug = IntensityAugmentation(
                clip=300, gamma_std=0.5, channel_wise=False, prob_gamma=0.95
            )
            self.fixed_blur = GaussianSmooth(sigma=0.5)

        # Frequency-domain downsampling
        self.freq_downsample = FrequencyDomainDownsample(
            volume_res=atlas_res,
            target_res=target_res,
            output_shape=output_shape,
            noise_std=0.01,
            prob_noise=0.65,
        )

        # Normalizer
        self.normalizer = ScaleIntensityRangePercentiles(
            lower=0, upper=100, b_min=0.0, b_max=1.0, clip=True
        )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] range using percentile scaling."""
        return self.normalizer(image)

    def generate_paired_data(
        self,
        hr_images: torch.Tensor,
        return_resolution: bool = False,
    ):
        """
        Generate paired LR-HR training data using frequency-domain downsampling.

        Args:
            hr_images: High-resolution input images, shape (B, C, D, H, W)
            return_resolution: If True, also returns resolution and thickness tensors

        Returns:
            If return_resolution=False:
                (lr_images, hr_augmented): Tuple of tensors, both (B, C, D, H, W)
            If return_resolution=True:
                (lr_images, hr_augmented, resolution, thickness): Tuple of 4 tensors
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # === HIGH-RESOLUTION PATH (Target) ===
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
            # Squeeze batch dim for GaussianSmooth, then unsqueeze
            lr_images_list = []
            for b in range(batch_size):
                blurred = self.fixed_blur(lr_images[b].unsqueeze(0).squeeze(0))
                lr_images_list.append(blurred.unsqueeze(0))
            lr_images = torch.stack(lr_images_list, dim=0)

        # Normalize both HR and LR to [0, 1]
        hr_augmented = self._normalize_image(hr_augmented)
        lr_images = self._normalize_image(lr_images)

        # Create brain mask to preserve background (optional)
        if self.preserve_background:
            brain_mask = (lr_images > self.background_threshold).float()
            background = lr_images * (1 - brain_mask)

        # Sample resolution and apply frequency-domain downsampling
        resolution = None
        thickness = None

        if self.randomise_res:
            # Sample random acquisition parameters
            resolution, thickness = self.res_sampler(batch_size)
            resolution = resolution.to(device)
            thickness = thickness.to(device)

            # Apply frequency-domain downsampling
            lr_images = self.freq_downsample(lr_images, resolution, thickness)

            # Restore clean background (remove FFT artifacts from background)
            if self.preserve_background:
                lr_images = lr_images * brain_mask + background
        else:
            # Use fixed atlas resolution
            resolution = torch.tensor(
                [self.atlas_res] * batch_size, dtype=torch.float32, device=device
            )
            thickness = resolution.clone()

        # lr_images = self._normalize_image(lr_images)

        # Final clipping to ensure [0, 1] range
        if self.clip_to_unit_range:
            lr_images = torch.clamp(lr_images, 0.0, 1.0)
            hr_augmented = torch.clamp(hr_augmented, 0.0, 1.0)
        
        if return_resolution:
            return lr_images, hr_augmented, resolution, thickness
        else:
            return lr_images, hr_augmented


class GeneratorDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that generates LR-HR pairs on-the-fly using frequency-domain downsampling.

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
    Create a dataset for training/validation using frequency-domain downsampling.

    This function has the same interface as data.py's create_dataset,
    allowing easy switching between the two implementations.

    Args:
        image_paths: List of paths to high-resolution NIfTI images
        generator: HRLRDataGenerator instance for creating LR-HR pairs
        target_shape: Target spatial shape [D, H, W] for cropping/padding
        target_spacing: Target voxel spacing [x, y, z] in mm for resampling
        use_cache: If True, use MONAI CacheDataset for faster loading
        return_resolution: If True, dataset returns resolution/thickness info
        is_training: If True, use random crops; if False, use center crops

    Returns:
        torch.utils.data.Dataset: Dataset that yields LR-HR pairs

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
    """
    # Create data dictionary for MONAI
    data_dicts = [{"image": img_path} for img_path in image_paths]

    # Build preprocessing transforms
    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
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
            transforms.append(
                CenterSpatialCropd(keys=["image"], roi_size=target_shape)
            )
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
