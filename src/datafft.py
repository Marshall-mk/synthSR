"""
Enhanced Data loading and generation for SynthSR training.
Includes comprehensive k-space artifact simulation and Physics-based PSF blurring.
"""

import torch
import torch.fft
import numpy as np
import math
from typing import Optional, List, Union, Tuple

from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, CropForegroundd, RandSpatialCropd, SpatialPadd,
    ToTensord, GaussianSmooth, ScaleIntensityRangePercentiles,
    CenterSpatialCropd,
)

from .domain_rand import (
    SampleResolution, RandomSpatialDeformation,
    BiasFieldCorruption, IntensityAugmentation,
)

# --- 1. User's Blur Sigma Calculation ---

def blurring_sigma_for_downsampling(
    current_res: Union[torch.Tensor, np.ndarray, List[float]],
    downsample_res: Union[torch.Tensor, np.ndarray, List[float]],
    thickness: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    mult_coef: float = 0.42,
) -> torch.Tensor:
    """
    Compute Gaussian blur sigma for anti-aliasing before downsampling.
    
    Formula: sigma = mult_coef * max(thickness, downsample_res) / current_res
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

# --- 2. Dynamic Gaussian Blur Helper ---

def apply_dynamic_gaussian_blur(img: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Applies a separable 3D Gaussian blur with dynamic sigmas per dimension.
    
    Args:
        img: Input tensor (C, D, H, W)
        sigma: Sigma values for [D, H, W] dimensions
    """
    device = img.device
    channels = img.shape[0]
    
    # Process each spatial dimension (Separable convolution: Blur Z -> Blur Y -> Blur X)
    for i, dim in enumerate([1, 2, 3]): # D, H, W correspond to dims 1, 2, 3
        s = sigma[i].item()
        
        # Skip if sigma is negligible
        if s < 0.1:
            continue
            
        # Calculate kernel size (3 sigmas either side)
        kernel_size = int(math.ceil(3 * s) * 2 + 1)
        # Ensure odd size
        if kernel_size % 2 == 0: kernel_size += 1
            
        # Create 1D Gaussian Kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / s) ** 2)
        kernel = kernel / kernel.sum()
        
        # Reshape for conv1d: (Out_C, In_C/Groups, K) -> (C, 1, K)
        kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        
        # We need to reshape img to use conv1d on the specific dimension
        # Move target dim to last, flatten others
        if i == 0: # Depth (D)
            # (C, D, H, W) -> (C, H, W, D) -> (C*H*W, 1, D)
            permute_shape = (0, 2, 3, 1) 
            inv_permute = (0, 3, 1, 2)
        elif i == 1: # Height (H)
            permute_shape = (0, 1, 3, 2)
            inv_permute = (0, 1, 3, 2)
        else: # Width (W)
            permute_shape = (0, 1, 2, 3) # Already last
            inv_permute = (0, 1, 2, 3)
            
        img_perm = img.permute(*permute_shape)
        orig_shape = img_perm.shape
        
        # Reshape for conv1d: (Batch, Channel, Length)
        # Here we treat (C, H, W) as "Batch" and D as "Length"
        flat_img = img_perm.reshape(-1, 1, orig_shape[-1])
        
        # Convolve
        # Padding = same
        blurred_flat = torch.nn.functional.conv1d(
            flat_img, 
            kernel[0:1], # Shared kernel
            padding=kernel_size//2
        )
        
        # Restore shape
        img = blurred_flat.reshape(orig_shape).permute(*inv_permute)

    return img

# --- 3. Artifact Helpers (Motion, Spikes, Aliasing) ---

def apply_kspace_motion_ghosting(volume: torch.Tensor, axis: int, intensity: float = 0.5, num_ghosts: int = 2) -> torch.Tensor:
    k_space = torch.fft.fftn(volume, dim=(1, 2, 3))
    k_space = torch.fft.fftshift(k_space, dim=(1, 2, 3))
    dims = volume.shape[1:]
    phase_axis_len = dims[axis]
    indices = torch.arange(phase_axis_len, device=volume.device)
    phase_error = torch.exp(1j * intensity * torch.sin(2 * np.pi * num_ghosts * indices / phase_axis_len))
    view_shape = [1, 1, 1, 1]
    view_shape[axis + 1] = phase_axis_len
    phase_error = phase_error.view(*view_shape)
    k_space_corrupted = k_space * phase_error
    k_space_corrupted = torch.fft.ifftshift(k_space_corrupted, dim=(1, 2, 3))
    return torch.abs(torch.fft.ifftn(k_space_corrupted, dim=(1, 2, 3)))

def apply_kspace_spike(volume: torch.Tensor, intensity: float = 5.0) -> torch.Tensor:
    k_space = torch.fft.fftn(volume, dim=(1, 2, 3))
    C, D, H, W = volume.shape
    rd, rh, rw = torch.randint(0, D, (1,)), torch.randint(0, H, (1,)), torch.randint(0, W, (1,))
    spike_val = torch.max(torch.abs(k_space)) * intensity
    k_space[:, rd, rh, rw] += spike_val
    return torch.abs(torch.fft.ifftn(k_space, dim=(1, 2, 3)))

def apply_aliasing(volume: torch.Tensor, axis: int, fold_pct: float = 0.2) -> torch.Tensor:
    dims = list(volume.shape)
    spatial_axis = axis + 1
    original_size = dims[spatial_axis]
    shift = int(original_size * fold_pct / 2)
    wrapped = torch.roll(volume, shifts=shift, dims=spatial_axis) * 0.5 + \
              torch.roll(volume, shifts=-shift, dims=spatial_axis) * 0.5
    return (volume + wrapped) / 1.5


# --- 4. Main Simulator Class ---

class MRIArtifactSimulator(torch.nn.Module):
    """
    Comprehensive MRI Artifact Simulator.
    Pipeline:
    1. Physics-based Blurring (PSF/Slice Thickness) <- Your function applied here
    2. K-space corruptions (Motion, Spikes)
    3. Aliasing/Wrap
    4. Downsampling (Resolution loss)
    5. Noise
    """

    def __init__(
        self,
        volume_res: List[float],
        target_res: List[float],
        output_shape: List[int],
        prob_motion: float = 0.2,
        prob_spike: float = 0.1,
        prob_aliasing: float = 0.1,
        prob_noise: float = 0.95,
        noise_std: float = 0.01,
        motion_intensity: float = 0.5,
        spike_intensity: float = 2.0,
    ):
        super().__init__()
        self.volume_res = torch.tensor(volume_res, dtype=torch.float32)
        self.target_res = torch.tensor(target_res, dtype=torch.float32)
        self.output_shape = output_shape
        self.prob_motion = prob_motion
        self.prob_spike = prob_spike
        self.prob_aliasing = prob_aliasing
        self.prob_noise = prob_noise
        self.noise_std = noise_std
        self.motion_intensity = motion_intensity
        self.spike_intensity = spike_intensity

    def forward(
        self,
        image: torch.Tensor,
        acquisition_res: torch.Tensor,
        thickness: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size = image.shape[0]
        device = image.device
        outputs = []

        for b in range(batch_size):
            img = image[b]  # (C, D, H, W)
            acq_res = acquisition_res[b] if acquisition_res.ndim > 1 else acquisition_res
            acq_res = acq_res.to(device)
            
            if thickness is not None:
                thk = thickness[b] if thickness.ndim > 1 else thickness
                thk = thk.to(device)
            else:
                thk = acq_res

            # --- STEP 1: PSF Blurring (Slice Profile / Partial Volume) ---
            # Using your provided function
            sigma = blurring_sigma_for_downsampling(
                current_res=self.volume_res.to(device),
                downsample_res=acq_res,
                thickness=thk
            )
            
            # Apply the calculated blur BEFORE any downsampling or k-space corruption
            # This simulates the physical magnetization averaging over the voxel
            img = apply_dynamic_gaussian_blur(img, sigma)


            # --- STEP 2: K-Space Artifacts (Motion / Spike) ---
            if torch.rand(1).item() < self.prob_motion:
                axis = torch.randint(1, 3, (1,)).item() 
                img = apply_kspace_motion_ghosting(img, axis=axis, intensity=self.motion_intensity)

            if torch.rand(1).item() < self.prob_spike:
                img = apply_kspace_spike(img, intensity=self.spike_intensity)
                
            # --- STEP 3: Aliasing ---
            if torch.rand(1).item() < self.prob_aliasing:
                axis = torch.randint(1, 3, (1,)).item()
                img = apply_aliasing(img, axis=axis, fold_pct=0.15)

            # --- STEP 4: Resolution Reduction (FFT Downsample) ---
            factors = acq_res / self.volume_res.to(device)
            downsample_axis = torch.argmax(factors).item()
            factor = factors[downsample_axis].item()

            if factor > 1.1:
                # Downsample via FFT cropping (simulates acquisition matrix limit)
                original_shape = img.shape
                spatial_axis = downsample_axis + 1
                new_size = int(round(original_shape[spatial_axis] / factor))

                fft_volume = torch.fft.fftn(img, dim=(1, 2, 3))
                fft_volume = torch.fft.fftshift(fft_volume, dim=(1, 2, 3))

                center_idx = original_shape[spatial_axis] // 2
                crop_start = center_idx - new_size // 2
                crop_end = crop_start + new_size

                if downsample_axis == 0:
                    cropped_fft = fft_volume[:, crop_start:crop_end, :, :]
                elif downsample_axis == 1:
                    cropped_fft = fft_volume[:, :, crop_start:crop_end, :]
                else:
                    cropped_fft = fft_volume[:, :, :, crop_start:crop_end]

                cropped_fft = torch.fft.ifftshift(cropped_fft, dim=(1, 2, 3))
                img = torch.real(torch.fft.ifftn(cropped_fft, dim=(1, 2, 3)))

                # Upsample back (Nearest Neighbor) to maintain "Blocky/Stair-step" look
                if list(img.shape[1:]) != self.output_shape:
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=self.output_shape,
                        mode="nearest", 
                    ).squeeze(0)

            # --- STEP 5: Noise ---
            if self.prob_noise > 0 and torch.rand(1).item() < self.prob_noise:
                n1 = torch.randn_like(img) * self.noise_std
                n2 = torch.randn_like(img) * self.noise_std
                img = torch.sqrt((img + n1)**2 + n2**2)

            outputs.append(img.unsqueeze(0))

        return torch.cat(outputs, dim=0)

class HRLRDataGenerator:
    """
    Enhanced Generator that orchestrates the High-Res to Low-Res degradation pipeline.

    It combines:
    1. Spatial Domain Augmentations (Deformation, Bias Field, Intensity)
    2. MRI Physics Simulation (via MRIArtifactSimulator)

    Args:
        atlas_res: Resolution of input HR images in mm [x, y, z]
        target_res: Target output resolution in mm [x, y, z]
        output_shape: Output spatial shape [D, H, W]
        
        # --- Probability Controls ---
        prob_motion: Probability of simulated patient motion (ghosting)
        prob_spike: Probability of k-space spikes (zipper artifact)
        prob_aliasing: Probability of fold-over aliasing
        prob_bias_field: Probability of RF inhomogeneity
        prob_noise: Probability of adding Rician noise
        prob_deform: Probability of spatial elastic deformation
        
        # --- Other Parameters ---
        min_resolution: Minimum resolution to sample [x, y, z]
        max_res_aniso: Maximum slice thickness to sample [x, y, z]
        randomise_res: If True, randomizes resolution/thickness
        apply_intensity_aug: If True, applies gamma/contrast augmentation
        clip_to_unit_range: If True, clips output to [0, 1]
    """

    def __init__(
        self,
        atlas_res: list = [1.0, 1.0, 1.0],
        target_res: list = [1.0, 1.0, 1.0],
        output_shape: list = [128, 128, 128],
        # Probabilities
        prob_motion: float = 0.2,
        prob_spike: float = 0.05,
        prob_aliasing: float = 0.1,
        prob_bias_field: float = 0.5,
        prob_noise: float = 0.8,
        prob_deform: float = 0.8,
        # Resolution simulation
        min_resolution: list = [1.0, 1.0, 1.0],
        max_res_aniso: list = [9.0, 9.0, 9.0],
        randomise_res: bool = True,
        # Toggles
        apply_intensity_aug: bool = True,
        clip_to_unit_range: bool = True,
    ):
        self.atlas_res = atlas_res
        self.target_res = target_res
        self.output_shape = output_shape
        self.randomise_res = randomise_res
        self.apply_intensity_aug = apply_intensity_aug
        self.clip_to_unit_range = clip_to_unit_range
        
        # Probabilities for conditional execution
        self.prob_bias_field = prob_bias_field
        self.prob_deform = prob_deform

        # 1. Resolution Sampler
        if randomise_res:
            self.res_sampler = SampleResolution(
                min_resolution=min_resolution,
                max_res_iso=None,
                max_res_aniso=max_res_aniso,
                prob_iso=0.0, # Force anisotropic for slice stack simulation
                prob_min=0.05,
                return_thickness=True,
            )

        # 2. Spatial Deformation (Anatomy shape change)
        self.deformer = RandomSpatialDeformation(
            scaling_bounds=0.15,
            rotation_bounds=15.0,
            shearing_bounds=0.012,
            translation_bounds=False,
            elastic_sigma_range=(5.0, 7.0),
            elastic_magnitude_range=(100.0, 200.0),
            prob_deform=1.0, # We handle probability in generate_paired_data
        )

        # 3. Bias Field (RF Inhomogeneity)
        self.bias = BiasFieldCorruption(
            bias_field_std=0.3, bias_scale=0.025, prob=1.0 # Prob handled manually
        )

        # 4. Intensity Augmentation (Contrast)
        if apply_intensity_aug:
            self.intensity_aug = IntensityAugmentation(
                clip=300, gamma_std=0.5, channel_wise=False, prob_gamma=0.5
            )

        # 5. MRI Physics Simulator (The core engine)
        self.artifact_simulator = MRIArtifactSimulator(
            volume_res=atlas_res,
            target_res=target_res,
            output_shape=output_shape,
            prob_motion=prob_motion,
            prob_spike=prob_spike,
            prob_aliasing=prob_aliasing,
            prob_noise=prob_noise,
            noise_std=0.02, # Base noise level
            motion_intensity=0.5
        )

        # Normalization helper
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
        Generate paired LR-HR training data.
        
        Flow:
        HR Image -> Deform -> Bias -> Intensity -> [Simulator: Blur->Artifacts->Downsample->Noise] -> LR Image
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # === HIGH-RESOLUTION PATH (Target) ===
        hr_augmented = hr_images.clone()

        # === LOW-RESOLUTION PATH (Input) ===
        lr_images = hr_augmented.clone()

        # 1. Apply Spatial Deformations (Anatomy)
        # Applied to BOTH HR and LR so they align, or just LR? 
        # Typically in SR, we want HR and LR to be spatially aligned "ground truth".
        # So we deform both (or the input) before splitting.
        if torch.rand(1).item() < self.prob_deform:
            # We deform the base image, so HR and LR track the same anatomy
            lr_images = self.deformer(lr_images)
            hr_augmented = lr_images.clone() # HR ground truth is the deformed high-quality image

        # 2. Apply Bias Field 
        # Usually only on LR (input), as we want the network to remove it.
        if torch.rand(1).item() < self.prob_bias_field:
            lr_images = self.bias(lr_images)

        # 3. Apply Intensity Augmentation
        if self.apply_intensity_aug:
            lr_images = self.intensity_aug(lr_images)

        # Normalize to [0, 1] before physics simulation
        hr_augmented = self._normalize_image(hr_augmented)
        lr_images = self._normalize_image(lr_images)

        # 4. MRI Physics Simulation
        resolution = None
        thickness = None

        if self.randomise_res:
            # Sample random acquisition parameters
            resolution, thickness = self.res_sampler(batch_size)
            resolution = resolution.to(device)
            thickness = thickness.to(device)

            # Apply: Blur -> Ghosting -> Aliasing -> Downsampling -> Noise
            lr_images = self.artifact_simulator(lr_images, resolution, thickness)
        else:
            # Fixed resolution path
            resolution = torch.tensor([self.atlas_res] * batch_size, dtype=torch.float32, device=device)
            thickness = resolution.clone()
            lr_images = self.artifact_simulator(lr_images, resolution, thickness)

        # Final normalization to ensure strict [0, 1] range after noise addition
        lr_images = self._normalize_image(lr_images)

        if self.clip_to_unit_range:
            lr_images = torch.clamp(lr_images, 0.0, 1.0)
            hr_augmented = torch.clamp(hr_augmented, 0.0, 1.0)
        
        if return_resolution:
            return lr_images, hr_augmented, resolution, thickness
        else:
            return lr_images, hr_augmented


# --- Dataset Wrappers ---

class GeneratorDataset(torch.utils.data.Dataset):
    """Wrapper that applies the HRLRDataGenerator to a base dataset."""
    def __init__(self, base_dataset, generator, return_resolution):
        self.base_dataset = base_dataset
        self.generator = generator
        self.return_resolution = return_resolution

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        hr_image = data["image"]

        if hr_image.ndim == 3:
            hr_image = hr_image.unsqueeze(0)
        hr_image = hr_image.unsqueeze(0)  # (1, C, D, H, W)

        result = self.generator.generate_paired_data(
            hr_image,
            return_resolution=self.return_resolution,
        )

        if self.return_resolution:
            lr_image, hr_augmented, resolution, thickness = result
            return lr_image.squeeze(0), hr_augmented.squeeze(0), resolution.squeeze(0), thickness.squeeze(0)
        else:
            lr_image, hr_augmented = result
            return lr_image.squeeze(0), hr_augmented.squeeze(0)


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
    Creates the training dataset using HRLRDataGenerator.
    """
    data_dicts = [{"image": img_path} for img_path in image_paths]

    transforms = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
    ]

    if target_spacing is not None:
        transforms.append(Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"))

    if target_shape is not None:
        transforms.append(CropForegroundd(keys=["image"], source_key="image"))
        if is_training:
            transforms.append(RandSpatialCropd(keys=["image"], roi_size=target_shape, random_size=False))
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))
        else:
            transforms.append(CenterSpatialCropd(keys=["image"], roi_size=target_shape))
            transforms.append(SpatialPadd(keys=["image"], spatial_size=target_shape))

    transforms.append(ToTensord(keys=["image"]))
    transform = Compose(transforms)

    if use_cache:
        dataset = CacheDataset(data=data_dicts, transform=transform, cache_rate=1.0, num_workers=4)
    else:
        dataset = Dataset(data=data_dicts, transform=transform)

    return GeneratorDataset(dataset, generator, return_resolution)