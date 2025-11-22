"""
Domain randomization transforms for medical image synthesis.

This module implements advanced augmentation techniques specifically designed for
MRI super-resolution training, including spatial deformations, bias field corruption,
intensity augmentation, and realistic MRI acquisition simulation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union, List

from monai.transforms import (
    GaussianSmooth,
    Resize,
    RandGaussianNoise,
    Rand3DElastic,
    RandAffine,
)


class RandomSpatialDeformation(nn.Module):
    """
    Apply random spatial deformations to 3D volumes using affine and elastic transforms.

    Combines affine transformations (scaling, rotation, shearing, translation) with
    elastic deformations to create realistic anatomical variations. Optionally supports
    90-degree rotations for axis alignment variations.

    This transform is essential for training robust medical imaging models that can
    handle variations in patient positioning and anatomical differences.

    Args:
        scaling_bounds: Scaling factor bounds. If float, uses (1-bounds, 1+bounds).
                       If tuple, uses as (min, max). Default: 0.15 (±15% scaling)
        rotation_bounds: Maximum rotation angle in degrees for each axis. Default: 15.0
        shearing_bounds: Maximum shearing angle. Default: 0.012
        translation_bounds: Maximum translation in voxels. If False, no translation.
                           Default: False
        enable_90_rotations: If True, randomly apply 90° rotations along each axis
                            before affine transform. Useful for axis alignment variations.
                            Default: False
        elastic_sigma_range: Range for elastic deformation smoothness (sigma).
                            Lower = sharper deformations. Default: (5.0, 7.0)
        elastic_magnitude_range: Range for elastic deformation magnitude.
                                Higher = stronger deformations. Default: (100.0, 200.0)
        prob_deform: Probability of applying deformation. Default: 1.0 (always apply)

    Input Shape:
        (B, C, D, H, W) - Batch of 3D volumes

    Output Shape:
        (B, C, D, H, W) - Deformed volumes with same shape

    Example:
        >>> deformer = RandomSpatialDeformation(
        ...     scaling_bounds=0.15,
        ...     rotation_bounds=15,
        ...     enable_90_rotations=False,
        ...     prob_deform=0.95
        ... )
        >>> input_volume = torch.randn(2, 1, 128, 128, 128)
        >>> deformed = deformer(input_volume)
        >>> print(deformed.shape)  # (2, 1, 128, 128, 128)
    """

    def __init__(
        self,
        scaling_bounds: Union[float, Tuple[float, float]] = 0.15,
        rotation_bounds: float = 15.0,
        shearing_bounds: float = 0.012,
        translation_bounds: Union[bool, float] = False,
        enable_90_rotations: bool = False,
        elastic_sigma_range: Tuple[float, float] = (5.0, 7.0),
        elastic_magnitude_range: Tuple[float, float] = (100.0, 200.0),
        prob_deform: float = 1.0,
    ):
        super().__init__()
        self.prob_deform = prob_deform
        self.enable_90_rotations = enable_90_rotations

        if isinstance(scaling_bounds, float):
            scale_range = (1.0 - scaling_bounds, 1.0 + scaling_bounds)
        else:
            scale_range = scaling_bounds if scaling_bounds else None

        self.affine_transform = RandAffine(
            prob=1.0,
            rotate_range=(
                np.radians(rotation_bounds),
                np.radians(rotation_bounds),
                np.radians(rotation_bounds),
            )
            if rotation_bounds
            else None,
            scale_range=(
                (scale_range[0], scale_range[1]),
                (scale_range[0], scale_range[1]),
                (scale_range[0], scale_range[1]),
            )
            if scale_range
            else None,
            shear_range=(
                (-shearing_bounds, shearing_bounds),
                (-shearing_bounds, shearing_bounds),
                (-shearing_bounds, shearing_bounds),
            )
            if shearing_bounds
            else None,
            translate_range=(
                (-translation_bounds, translation_bounds),
                (-translation_bounds, translation_bounds),
                (-translation_bounds, translation_bounds),
            )
            if translation_bounds
            else None,
            padding_mode="border",
            mode="bilinear",
        )

        self.elastic_transform = Rand3DElastic(
            prob=1.0,
            sigma_range=elastic_sigma_range,
            magnitude_range=elastic_magnitude_range,
            padding_mode="border",
            mode="bilinear",
        )

    def forward(
        self, image: torch.Tensor, interpolation: str = "bilinear"
    ) -> torch.Tensor:
        """
        Apply random spatial deformations to input volumes.

        Args:
            image: Input tensor of shape (B, C, D, H, W)
            interpolation: Interpolation mode (not currently used, kept for API compatibility)

        Returns:
            Deformed tensor of shape (B, C, D, H, W)
        """
        if torch.rand(1).item() > self.prob_deform:
            return image

        batch_size = image.shape[0]
        outputs = []

        for b in range(batch_size):
            img = image[b : b + 1]

            # Apply 90-degree rotations if enabled
            if self.enable_90_rotations:
                for axis in range(3):
                    k = torch.randint(0, 4, (1,)).item()
                    if k > 0:
                        if axis == 0:  # Rotate in D-H plane
                            img = torch.rot90(img, k=k, dims=(2, 3))
                        elif axis == 1:  # Rotate in D-W plane
                            img = torch.rot90(img, k=k, dims=(2, 4))
                        else:  # Rotate in H-W plane
                            img = torch.rot90(img, k=k, dims=(3, 4))

            # Apply affine and elastic transforms
            img = self.affine_transform(img)
            img = self.elastic_transform(img)
            outputs.append(img)

        return torch.cat(outputs, dim=0)


class BiasFieldCorruption(nn.Module):
    """
    Simulate MRI bias field (intensity inhomogeneity) artifacts.

    MRI scans often exhibit smooth intensity variations across the image due to
    B1 field inhomogeneity. This transform simulates these artifacts by generating
    smooth multiplicative bias fields and applying them to the input images.

    The bias field is created at a lower resolution (controlled by bias_scale) and
    then upsampled to the image size, ensuring smoothness. This approach is
    computationally efficient while producing realistic bias field patterns.

    Args:
        bias_field_std: Standard deviation of the bias field coefficients.
                       Higher values create stronger bias field effects.
                       Default: 0.3
        bias_scale: Scale factor for bias field resolution relative to image size.
                   Smaller values create smoother bias fields.
                   For example, 0.025 on a 128³ image creates a 3³ bias field.
                   Default: 0.025
        prob: Probability of applying bias field corruption. Default: 0.98

    Input Shape:
        (B, C, D, H, W) - Batch of 3D volumes

    Output Shape:
        (B, C, D, H, W) - Corrupted volumes with bias field applied

    Example:
        >>> bias_corruptor = BiasFieldCorruption(
        ...     bias_field_std=0.3,
        ...     bias_scale=0.025,
        ...     prob=0.98
        ... )
        >>> input_volume = torch.randn(2, 1, 128, 128, 128)
        >>> corrupted = bias_corruptor(input_volume)
        >>> # Corrupted volume has smooth intensity variations
    """

    def __init__(
        self, bias_field_std: float = 0.3, bias_scale: float = 0.025, prob: float = 0.98
    ):
        super().__init__()
        self.bias_field_std = bias_field_std
        self.bias_scale = bias_scale
        self.prob = prob

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply bias field corruption to input volumes.

        Args:
            image: Input tensor of shape (B, C, D, H, W)

        Returns:
            Corrupted tensor of shape (B, C, D, H, W) with multiplicative bias field
        """
        if torch.rand(1).item() > self.prob:
            return image

        batch_size = image.shape[0]
        spatial_shape = image.shape[2:]
        device = image.device
        outputs = []

        for b in range(batch_size):
            img = image[b : b + 1]

            # Generate low-resolution bias field coefficients
            bias_shape = [max(1, int(s * self.bias_scale)) for s in spatial_shape]
            bias_coeffs = (
                torch.randn(1, 1, *bias_shape, device=device) * self.bias_field_std
            )

            # Upsample to image size (creates smooth bias field)
            resize_transform = Resize(spatial_size=spatial_shape, mode="trilinear")
            bias_field = resize_transform(bias_coeffs)

            # Convert to multiplicative field and apply
            bias_field = torch.exp(bias_field)
            img = img * bias_field
            outputs.append(img)

        return torch.cat(outputs, dim=0)


class IntensityAugmentation(nn.Module):
    """
    Apply intensity-based augmentations including clipping and gamma correction.

    Performs two main operations:
    1. **Clipping**: Removes extreme intensity outliers that could destabilize training
    2. **Gamma correction**: Applies power-law intensity transformations to simulate
       different contrast settings and acquisition protocols

    Gamma correction: I_out = I_in^(exp(γ)) where γ ~ N(0, gamma_std)
    - γ > 0: Darkens mid-tones (compression)
    - γ < 0: Brightens mid-tones (expansion)

    Args:
        clip: Clipping bounds. Options:
             - float: Clip to [0, clip]
             - tuple: Clip to [clip[0], clip[1]]
             - False: No clipping
             Default: 300
        gamma_std: Standard deviation of gamma parameter for gamma correction.
                  Gamma is sampled from N(0, gamma_std). Higher = stronger variation.
                  Default: 0.5
        channel_wise: If True, apply different gamma per channel. If False, same
                     gamma for all channels. Default: False
        prob_gamma: Probability of applying gamma correction. Default: 0.95

    Input Shape:
        (B, C, D, H, W) - Batch of 3D volumes

    Output Shape:
        (B, C, D, H, W) - Augmented volumes

    Example:
        >>> intensity_aug = IntensityAugmentation(
        ...     clip=300,
        ...     gamma_std=0.5,
        ...     prob_gamma=0.95
        ... )
        >>> input_volume = torch.randn(2, 1, 128, 128, 128).abs() * 100
        >>> augmented = intensity_aug(input_volume)
        >>> # Augmented volume has clipped values and modified contrast
    """

    def __init__(
        self,
        clip: Union[float, Tuple[float, float], bool] = 300,
        gamma_std: float = 0.5,
        channel_wise: bool = False,
        prob_gamma: float = 0.95,
    ):
        super().__init__()
        self.clip = clip
        self.gamma_std = gamma_std
        self.channel_wise = channel_wise
        self.prob_gamma = prob_gamma

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply intensity augmentations to input volumes.

        Args:
            image: Input tensor of shape (B, C, D, H, W)

        Returns:
            Augmented tensor of shape (B, C, D, H, W)
        """
        batch_size, n_channels = image.shape[:2]
        ndims = len(image.shape) - 2
        device = image.device

        # 1. Clip outliers
        if self.clip:
            if isinstance(self.clip, (int, float)):
                image = torch.clamp(image, 0, self.clip)
            else:
                image = torch.clamp(image, self.clip[0], self.clip[1])

        # 2. Gamma augmentation (power-law transform)
        if self.gamma_std > 0 and torch.rand(1).item() < self.prob_gamma:
            if self.channel_wise:
                # Different gamma per channel
                gamma = (
                    torch.randn(batch_size, n_channels, *([1] * ndims), device=device)
                    * self.gamma_std
                )
            else:
                # Same gamma for all channels
                gamma = (
                    torch.randn(batch_size, 1, *([1] * ndims), device=device)
                    * self.gamma_std
                )

            # Apply power transform: I^(exp(gamma))
            image = torch.pow(image.clamp(min=1e-7), torch.exp(gamma))

        return image


class SampleResolution(nn.Module):
    """
    Sample random acquisition resolutions for MRI simulation.

    Simulates different MRI acquisition protocols by randomly sampling voxel resolutions.
    Supports both isotropic (same resolution in all directions) and anisotropic
    (different resolutions, e.g., high in-plane, low through-plane) acquisitions.

    This is crucial for training models that must handle diverse acquisition protocols,
    such as:
    - T1 scans: typically 1×1×1mm isotropic
    - T2 scans: often 0.5×0.5×3mm anisotropic
    - Clinical scans: highly variable (1×1×5mm common)

    Args:
        min_resolution: Minimum resolution (highest quality) in mm for each axis.
                       Example: [1.0, 1.0, 1.0]
        max_res_iso: Maximum isotropic resolution (lowest quality) in mm.
                    Example: [1.0, 1.0, 1.0] for 1mm isotropic
                    Can be None if only anisotropic is used.
        max_res_aniso: Maximum anisotropic resolution in mm for each axis.
                      Example: [9.0, 9.0, 9.0]
                      Can be None if only isotropic is used.
        prob_iso: Probability of sampling isotropic resolution (vs anisotropic).
                 Default: 0.05 (95% anisotropic, 5% isotropic)
        prob_min: Probability of using minimum (highest quality) resolution.
                 Default: 0.05
        return_thickness: If True, also returns slice thickness (can differ from resolution).
                         Default: True

    Returns:
        If return_thickness=False:
            resolution: Tensor of shape (batch_size, 3) with sampled resolutions
        If return_thickness=True:
            (resolution, thickness): Tuple of tensors, both shape (batch_size, 3)

    Example:
        >>> res_sampler = SampleResolution(
        ...     min_resolution=[1.0, 1.0, 1.0],
        ...     max_res_iso=[1.0, 1.0, 1.0],
        ...     max_res_aniso=[9.0, 9.0, 9.0],
        ...     prob_iso=0.02,
        ...     prob_min=0.1
        ... )
        >>> resolution, thickness = res_sampler(batch_size=4)
        >>> print(resolution.shape)  # (4, 3)
        >>> # Example: [[1.0, 1.0, 5.2], [1.0, 1.0, 7.8], ...]
    """

    def __init__(
        self,
        min_resolution: List[float],
        max_res_iso: Optional[List[float]] = None,
        max_res_aniso: Optional[List[float]] = None,
        prob_iso: float = 0.05,
        prob_min: float = 0.05,
        return_thickness: bool = True,
    ):
        super().__init__()
        self.min_res = torch.tensor(min_resolution, dtype=torch.float32)
        self.max_res_iso = (
            torch.tensor(max_res_iso, dtype=torch.float32) if max_res_iso else None
        )
        self.max_res_aniso = (
            torch.tensor(max_res_aniso, dtype=torch.float32) if max_res_aniso else None
        )
        self.prob_iso = prob_iso
        self.prob_min = prob_min
        self.return_thickness = return_thickness
        self.n_dims = len(min_resolution)

        assert (max_res_iso is not None) or (max_res_aniso is not None), (
            "At least one of max_res_iso or max_res_aniso must be provided"
        )

    def forward(
        self, batch_size: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample random resolutions for a batch.

        Args:
            batch_size: Number of resolution samples to generate

        Returns:
            If return_thickness=False: resolution tensor of shape (batch_size, 3)
            If return_thickness=True: tuple of (resolution, thickness), both (batch_size, 3)
        """
        device = self.min_res.device

        # Determine which samples are isotropic vs anisotropic
        if (self.max_res_iso is not None) and (self.max_res_aniso is not None):
            use_iso = torch.rand(batch_size, device=device) < self.prob_iso
        elif self.max_res_iso is not None:
            use_iso = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            use_iso = torch.zeros(batch_size, dtype=torch.bool, device=device)

        resolution = torch.zeros(batch_size, self.n_dims, device=device)

        for b in range(batch_size):
            if use_iso[b]:
                # Isotropic: same resolution for all dimensions
                res_val = (
                    torch.rand(1, device=device) * (self.max_res_iso - self.min_res)
                    + self.min_res
                )
                resolution[b] = res_val[0]
            else:
                # Anisotropic: one dimension has high res, others have low res
                high_res_dim = torch.randint(0, self.n_dims, (1,), device=device).item()
                for d in range(self.n_dims):
                    if d == high_res_dim:
                        resolution[b, d] = (
                            torch.rand(1, device=device)
                            * (self.max_res_aniso[d] - self.min_res[d])
                            + self.min_res[d]
                        )
                    else:
                        resolution[b, d] = self.min_res[d]

        # Apply minimum resolution override
        use_min = torch.rand(batch_size, device=device) < self.prob_min
        resolution[use_min] = self.min_res.unsqueeze(0).expand(use_min.sum(), -1)

        if self.return_thickness:
            # Sample slice thickness (can be >= resolution)
            thickness = torch.zeros_like(resolution)
            for b in range(batch_size):
                for d in range(self.n_dims):
                    thickness[b, d] = (
                        torch.rand(1, device=device)
                        * (resolution[b, d] - self.min_res[d])
                        + self.min_res[d]
                    )
            return resolution, thickness
        else:
            return resolution


def blurring_sigma_for_downsampling(
    current_res: Union[torch.Tensor, np.ndarray, List[float]],
    downsample_res: Union[torch.Tensor, np.ndarray, List[float]],
    thickness: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    mult_coef: float = 0.42,
) -> torch.Tensor:
    """
    Compute Gaussian blur sigma for anti-aliasing before downsampling.

    When downsampling medical images, anti-aliasing blur is essential to prevent
    aliasing artifacts. This function computes the appropriate Gaussian blur sigma
    based on the downsampling factor, following standard signal processing principles.

    The formula accounts for slice thickness (which can differ from voxel spacing)
    to accurately simulate MRI acquisition physics.

    Formula:
        sigma = mult_coef * max(thickness, downsample_res) / current_res

    Args:
        current_res: Current voxel resolution in mm. Shape: (3,) or (batch_size, 3)
        downsample_res: Target downsampled resolution in mm. Shape: (3,) or (batch_size, 3)
        thickness: Slice thickness in mm (can be > downsample_res).
                  If None, uses downsample_res. Shape: (3,) or (batch_size, 3)
        mult_coef: Multiplier coefficient for sigma calculation.
                  Default: 0.42 (empirically determined for medical imaging)

    Returns:
        Gaussian blur sigma for each dimension. Shape matches input shapes.

    Example:
        >>> current_res = torch.tensor([1.0, 1.0, 1.0])  # 1mm isotropic
        >>> downsample_res = torch.tensor([1.0, 1.0, 5.0])  # 5mm in z
        >>> thickness = torch.tensor([1.0, 1.0, 5.0])
        >>> sigma = blurring_sigma_for_downsampling(current_res, downsample_res, thickness)
        >>> print(sigma)  # tensor([0.42, 0.42, 2.10]) - more blur in z direction
    """
    device = None
    if isinstance(current_res, torch.Tensor):
        device = current_res.device
    elif isinstance(downsample_res, torch.Tensor):
        device = downsample_res.device
    elif isinstance(thickness, torch.Tensor):
        device = thickness.device

    # Convert all inputs to tensors
    if not isinstance(current_res, torch.Tensor):
        current_res = torch.tensor(current_res, dtype=torch.float32, device=device)
    if not isinstance(downsample_res, torch.Tensor):
        downsample_res = torch.tensor(
            downsample_res, dtype=torch.float32, device=device
        )

    if thickness is None:
        thickness = downsample_res
    elif not isinstance(thickness, torch.Tensor):
        thickness = torch.tensor(thickness, dtype=torch.float32, device=device)

    # Use maximum of thickness and resolution for conservative blurring
    max_res = torch.maximum(thickness, downsample_res)
    sigma = mult_coef * max_res / current_res
    return sigma


class MimicAcquisition(nn.Module):
    """
    Simulate realistic MRI acquisition pipeline with resolution degradation.

    Mimics the complete MRI acquisition process:
    1. **Anti-aliasing blur**: Gaussian smoothing to prevent aliasing artifacts
    2. **Downsampling**: Reduce resolution to simulate lower-quality acquisition
    3. **Noise injection**: Add Gaussian noise to simulate scanner noise
    4. **Upsampling**: Restore to target shape (simulates image reconstruction)

    This pipeline creates realistic low-resolution MRI images that match the statistical
    properties of actual clinical scans, which is essential for training super-resolution
    models that generalize to real data.

    Args:
        volume_res: Original volume resolution in mm [x, y, z]. Example: [1.0, 1.0, 1.0]
        target_res: Target output resolution in mm [x, y, z]. Example: [1.0, 1.0, 1.0]
        output_shape: Output spatial shape [D, H, W]. Example: [128, 128, 128]
        noise_std: Standard deviation of Gaussian noise to add. Default: 0.01
        prob_noise: Probability of adding noise. Default: 0.95
        build_dist_map: If True, also return a distance map (not currently used).
                       Default: False

    Input:
        image: Tensor of shape (B, C, D, H, W) - high-resolution input
        acquisition_res: Tensor of shape (B, 3) or (3,) - target acquisition resolution
        thickness: Optional tensor of shape (B, 3) or (3,) - slice thickness

    Output:
        If build_dist_map=False:
            Degraded tensor of shape (B, C, *output_shape)
        If build_dist_map=True:
            Tuple of (degraded tensor, distance map)

    Example:
        >>> mimic = MimicAcquisition(
        ...     volume_res=[1.0, 1.0, 1.0],
        ...     target_res=[1.0, 1.0, 1.0],
        ...     output_shape=[128, 128, 128],
        ...     noise_std=0.01
        ... )
        >>> hr_image = torch.randn(2, 1, 128, 128, 128)
        >>> acq_res = torch.tensor([[1.0, 1.0, 5.0], [1.0, 1.0, 3.0]])  # Anisotropic
        >>> thickness = torch.tensor([[1.0, 1.0, 5.0], [1.0, 1.0, 3.0]])
        >>> lr_image = mimic(hr_image, acq_res, thickness)
        >>> # lr_image simulates low-res acquisition, upsampled to (2, 1, 128, 128, 128)
    """

    def __init__(
        self,
        volume_res: List[float],
        target_res: List[float],
        output_shape: List[int],
        noise_std: float = 0.01,
        prob_noise: float = 0.95,
        build_dist_map: bool = False,
    ):
        super().__init__()
        self.volume_res = torch.tensor(volume_res, dtype=torch.float32)
        self.target_res = torch.tensor(target_res, dtype=torch.float32)
        self.output_shape = output_shape
        self.noise_std = noise_std
        self.prob_noise = prob_noise
        self.build_dist_map = build_dist_map
        self.noise_transform = RandGaussianNoise(
            prob=prob_noise, mean=0.0, std=noise_std
        )

    def forward(
        self,
        image: torch.Tensor,
        acquisition_res: torch.Tensor,
        thickness: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply MRI acquisition simulation to input volumes.

        Args:
            image: Input high-resolution tensor of shape (B, C, D, H, W)
            acquisition_res: Target acquisition resolution, shape (B, 3) or (3,)
            thickness: Optional slice thickness, shape (B, 3) or (3,)
                      If None, uses acquisition_res

        Returns:
            If build_dist_map=False:
                Simulated low-res acquisition of shape (B, C, *output_shape)
            If build_dist_map=True:
                Tuple of (simulated acquisition, distance map)
        """
        batch_size = image.shape[0]
        device = image.device
        outputs = []

        for b in range(batch_size):
            img = image[b : b + 1]
            acq_res = (
                acquisition_res[b] if acquisition_res.ndim > 1 else acquisition_res
            )
            thick = (
                thickness[b]
                if thickness is not None and thickness.ndim > 1
                else acq_res
            )

            # 1. Anti-aliasing blur before downsampling
            sigma = blurring_sigma_for_downsampling(
                self.volume_res.to(device),
                acq_res.to(device),
                thick.to(device) if thick is not None else None,
                mult_coef=0.42,
            )
            if (sigma > 0.1).any():
                smooth = GaussianSmooth(sigma=sigma.cpu().numpy().tolist())
                img = smooth(img)

            # 2. Downsample to simulated acquisition resolution
            factor = acq_res.to(device) / self.volume_res.to(device)
            original_size = torch.tensor(
                img.shape[2:], dtype=torch.float32, device=device
            )
            new_size = (original_size / factor).long().clamp(min=1).tolist()
            resize_down = Resize(spatial_size=new_size, mode="trilinear")
            img = resize_down(img)

            # 3. Add Gaussian noise (simulates scanner noise)
            if self.noise_std > 0:
                img = self.noise_transform(img)

            # 4. Upsample back to target shape (simulates reconstruction)
            resize_up = Resize(spatial_size=self.output_shape, mode="trilinear")
            img = resize_up(img)
            outputs.append(img)

        result = torch.cat(outputs, dim=0)

        if self.build_dist_map:
            return result, torch.ones_like(result)
        else:
            return result
