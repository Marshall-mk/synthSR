"""
Data loading and generation for SynthSR training.
Includes comprehensive k-space artifact simulation and Physics-based PSF blurring.

This module simulates the MRI acquisition pipeline to generate synthetic Low-Resolution (LR) images
from High-Resolution (HR) ground truth. It models the physical degradation process including:
1.  **Slice Profile & Point Spread Function (PSF):** The physical blurring caused by RF excitation profiles and signal decay.
2.  **B1 Field Inhomogeneity:** Bias fields that cause smooth intensity variations.
3.  **Contrast Variation:** Gamma correction to simulate different T1/T2 weightings.
4.  **K-Space Artifacts:** Motion ghosting and spikes in the frequency domain.
5.  **Sampling Limits:** Resolution loss via FFT cropping (simulating limited k-space acquisition).
"""
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Union, Tuple

from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, CropForegroundd, RandSpatialCropd, SpatialPadd,
    ToTensord, ScaleIntensityRangePercentiles,
    CenterSpatialCropd, Resize, RandAffined,
)



class SliceProfilePhysics(nn.Module):
    """
    Simulates the physical blurring caused by the MRI scanner's slice selection profile and in-plane sampling.

    **MRI Physical Representation:**
    In a real MRI scanner, a 2D slice is not a perfect geometric plane. It has a thickness determined
    by the Radio Frequency (RF) excitation pulse.
    - **Through-Plane (Slice Direction):** The RF pulse excites spins within a "slab." The sensitivity profile
      across this slab is rarely a perfect rectangle (Boxcar). It is often trapezoidal or Gaussian due to
      hardware limits on the RF pulse duration (truncated Sinc pulses). This causes "Partial Volume Effects"
      where signal from adjacent tissues bleeds into the slice.
    - **In-Plane (Phase/Frequency Directions):** Blurring occurs due to T2* relaxation during readout
      and finite sampling windows, typically modeled as a Point Spread Function (PSF).

    This class replaces generic Gaussian blurring with physically programmable kernels to accurately
    model these distinct behaviors.

    Args:
        profile_type (str): The shape of the slice sensitivity profile.
            - 'boxcar': Ideal rectangular profile (perfect slice selection).
            - 'gaussian': Standard approximation (often used in simple simulations).
            - 'trapezoid': Realistic profile for most clinical scanners (flat top with fading edges).
        edge_width (float): For 'trapezoid', the fraction of the slice thickness that is the "slope"
            (transition region). Represents the imperfect sharp cutoff of the RF pulse.
    """
    
    def __init__(self, profile_type='trapezoid', edge_width=0.1):
        """
        Args:
            profile_type: 'gaussian', 'boxcar' (ideal), or 'trapezoid' (realistic).
            edge_width: For trapezoid, how much of the slice is the "slope" (0.0-0.5).
                        0.1 means 10% on left and 10% on right are fading out.
        """
        super().__init__()
        self.profile_type = profile_type
        self.edge_width = edge_width

    def get_slice_kernel(self, thickness_mm, current_res_mm, device):
        """
        Generates the 1D convolution kernel representing the slice sensitivity profile.
        
        Args:
            thickness_mm (float): The target slice thickness to simulate.
            current_res_mm (float): The current resolution of the input image.
            device (torch.device): Device to create tensors on.
            
        Returns:
            torch.Tensor: Normalized 1D kernel.
        """
        
        # Calculate kernel size in voxels
        # We need enough support to capture the profile
        scale = thickness_mm / current_res_mm
        kernel_size = int(math.ceil(scale * 3)) 
        if kernel_size % 2 == 0: kernel_size += 1
        
        grid = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
        
        # Normalize grid relative to slice thickness (0.5 = half thickness)
        x = grid / scale 

        if self.profile_type == 'boxcar':
            # Ideal rectangular profile: 1 inside [-0.5, 0.5], 0 outside
            kernel = (x.abs() <= 0.5).float()
            
        elif self.profile_type == 'gaussian':
            # Standard approximation (FWHM = thickness)
            # sigma corresponding to FWHM=1 is 1 / 2.355 = 0.4246
            sigma = 0.4246
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            
        elif self.profile_type == 'trapezoid':
            # Realistic profile: Flat top with sloping edges
            # Width of the flat top
            flat_width = 0.5 - self.edge_width
            
            # Mask for flat region
            flat_mask = (x.abs() <= flat_width).float()
            
            # Mask for slopes
            slope_mask = ((x.abs() > flat_width) & (x.abs() <= 0.5)).float()
            
            # Linear decay on slopes
            # Dist from edge start / edge width
            slope_val = 1.0 - (x.abs() - flat_width) / self.edge_width
            
            kernel = flat_mask + slope_mask * slope_val
            
        else:
            raise ValueError(f"Unknown profile: {self.profile_type}")

        # Energy conservation (area under curve must be 1)
        return kernel / kernel.sum()

    def forward(self, img, resolution, thickness):
        """
        Applies the physics-based blurring to the input volume.

        This method identifies the "Slice Select" direction (the one with the lowest resolution/highest thickness)
        and applies the specific Slice Profile kernel. For the other two directions (In-Plane), it applies
        a standard PSF blur.

        Args:
            img (torch.Tensor): Input image tensor (C, D, H, W).
            resolution (torch.Tensor): Current voxel size of the input [res_D, res_H, res_W].
            thickness (torch.Tensor): Target slice thickness to simulate [thick_D, thick_H, thick_W].
        """
        device = img.device
        channels = img.shape[0]
        
        # Identify the slice dimension (the one with largest thickness/resolution ratio)
        # Usually MRI stacks are anisotropic, so the "thick" axis is the slice axis.
        factors = thickness / resolution
        slice_dim_idx = torch.argmax(factors).item()
        
        # We process dimensions 1, 2, 3 (D, H, W)
        for i, dim in enumerate([1, 2, 3]):
            
            # CASE A: Through-Plane (Slice Selection Axis)
            # Apply the specific slice profile (Boxcar/Trapezoid/Gaussian)
            if i == slice_dim_idx and factors[i] > 1.1: # Only if actually thick
                kernel = self.get_slice_kernel(thickness[i], resolution[i], device)
                
            # CASE B: In-Plane (Frequency/Phase Encoding Axes)
            # Apply standard PSF (Gaussian) due to T2* decay and sampling
            else:
                # Use standard Gaussian approximation for in-plane PSF
                # Sigma is small (approx 0.5-0.8 pixels) for in-plane
                sigma = 0.42 * (resolution[i] / resolution[i]) # ~0.42 pixels
                
                k_size = 5
                k_grid = torch.arange(k_size, device=device) - k_size//2
                kernel = torch.exp(-0.5 * (k_grid / sigma) ** 2)
                kernel = kernel / kernel.sum()

            # --- Convolve ---
            # Reshape kernel for conv1d: (C, 1, K)
            kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
            padding = kernel.shape[-1] // 2
            
            # Permute dimensions to apply 1D conv on the current axis
            if i == 0:   # D
                img_in = img.permute(0, 2, 3, 1) # (C, H, W, D)
            elif i == 1: # H
                img_in = img.permute(0, 1, 3, 2) # (C, D, W, H)
            else:        # W
                img_in = img # (C, D, H, W) is already fine for W if flattened differently?
                # Actually conv1d operates on the last dim.
                # So for W, we need input (C, D, H, W) -> flatten -> (Batch, C, W)
                img_in = img.permute(0, 1, 2, 3) # No change needed relative to "last dim" logic
            
            # Flatten non-active dims into batch
            shape_before = img_in.shape
            # Combine all dims except the last one (the active one)
            img_flat = img_in.reshape(-1, 1, shape_before[-1]) 
            
            # Apply Convolution
            img_filtered = F.conv1d(img_flat, kernel[0:1], padding=padding)
            
            # Un-flatten and un-permute
            img_out = img_filtered.view(shape_before)
            
            if i == 0:
                img = img_out.permute(0, 3, 1, 2)
            elif i == 1:
                img = img_out.permute(0, 1, 3, 2)
            else:
                img = img_out # Already correct
                
        return img


class BiasFieldCorruption(nn.Module):
    """
    Simulates MRI bias field (B1 inhomogeneity) artifacts.

    **MRI Physical Representation:**
    In MRI, the transmit/receive coils are not perfectly uniform. The sensitivity of the
    Radio Frequency (RF) coils varies across space, especially in older scanners or with
    surface coils. This causes low-frequency intensity variations where some parts of the
    brain appear brighter or darker than others, despite having the same tissue type.
    This is often called "intensity non-uniformity" (INU) or "shading."

    The bias field is modeled as a multiplicative field that varies smoothly over the image volume.

    Args:
        bias_field_std (float): Standard deviation of the bias field coefficients.
            Higher values create stronger shading effects (simulating poorer coil homogeneity).
        bias_scale (float): Scale factor for the bias field resolution.
            Controls the "frequency" of the shading. Smaller values mean very smooth,
            gradual shading (typical of body coils); larger values allow for more localized
            variations (typical of multi-channel surface coils).
        prob (float): Probability of applying this corruption.

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
        Applies a multiplicative bias field to the input volume.

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
            # Resize expects input without batch dimension (C, D, H, W)
            bias_field = resize_transform(bias_coeffs.squeeze(0)).unsqueeze(0)

            # Convert to multiplicative field and apply
            bias_field = torch.exp(bias_field)
            img = img * bias_field
            outputs.append(img)

        return torch.cat(outputs, dim=0)


class IntensityAugmentation(nn.Module):
    """
    Simulates variations in MRI contrast mechanisms and sensor dynamics.

    **MRI Physical Representation:**
    1. **Clipping:** Simulates the dynamic range limits of the MRI receiver/ADC (Analog-to-Digital Converter).
       Extremely high signal intensities (e.g., from fat or flow artifacts) can saturate the sensor.
    2. **Gamma Correction:** Simulates variations in tissue contrast (T1/T2 weighting).
       Different pulse sequences (TE, TR settings) produce different contrast curves.
       A power-law transform approximates these non-linear relationships between proton density
       and final pixel intensity.
       - Gamma < 1: Simulates images with brighter mid-tones (e.g., PD-weighted).
       - Gamma > 1: Simulates images with darker mid-tones (higher contrast).

    Gamma correction: I_out = I_in^(exp(γ)) where γ ~ N(0, gamma_std)

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
    Simulates the selection of MRI acquisition protocols (Field of View and Matrix Size).

    **MRI Physical Representation:**
    MRI scanners are configured by technicians to acquire data at specific resolutions.
    - **Isotropic:** High-resolution 3D scans (e.g., MP-RAGE) often have 1x1x1 mm voxels.
    - **Anisotropic:** Fast clinical 2D scans (e.g., T2-weighted turbo spin echo) typically
      have high in-plane resolution (e.g., 0.5 mm) but thick slices (e.g., 5.0 mm) to save time.

    This class randomizes these parameters to train the model to handle diverse
    clinical scenarios, from high-quality research scans to rapid emergency protocols.

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
    Used for generic anti-aliasing approximations when physics-based SliceProfile is disabled.
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
    """
    Simulates motion artifacts (ghosting) in K-space.

    **MRI Physical Representation:**
    Patient movement during the acquisition (especially during the Phase Encoding step) causes
    positional inconsistencies in the frequency data. This manifests as "ghosts" or faint copies
    of the anatomy propagated along the Phase Encoding direction.
    
    This function applies a phase error modulation in K-space to mathematically reproduce this effect.
    """
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
    """
    Simulates RF spikes (zipper artifacts).

    **MRI Physical Representation:**
    Stray radio frequency (RF) interference (e.g., from a light bulb or unshielded equipment)
    can appear as a high-intensity "spike" at a specific point in K-space.
    When reconstructed via Inverse FFT, a single point in K-space transforms into a 
    periodic stripe or "herringbone" pattern across the entire image.
    """
    k_space = torch.fft.fftn(volume, dim=(1, 2, 3))
    C, D, H, W = volume.shape
    rd, rh, rw = torch.randint(0, D, (1,)), torch.randint(0, H, (1,)), torch.randint(0, W, (1,))
    spike_val = torch.max(torch.abs(k_space)) * intensity
    k_space[:, rd, rh, rw] += spike_val
    return torch.abs(torch.fft.ifftn(k_space, dim=(1, 2, 3)))

def apply_aliasing(volume: torch.Tensor, axis: int, fold_pct: float = 0.2) -> torch.Tensor:
    """
    Simulates wrap-around aliasing (fold-over artifacts).

    **MRI Physical Representation:**
    If the Field of View (FOV) is smaller than the anatomy in the Phase Encoding direction,
    signal from outside the FOV "wraps around" to the opposite side of the image.
    This is common in abdominal or shoulder MRI where the body extends beyond the selected box.
    """
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
    The physics engine that orchestrates the degradation pipeline.

    **Simulation Pipeline:**
    1.  **Slice Profile (Physics-Based):** Applies realistic slice blurring (Trapezoidal/Boxcar) before downsampling.
    2.  **K-Space Corruptions:** Transforms data to frequency domain to add Motion Ghosts and RF Spikes.
    3.  **Aliasing:** Simulates FOV wrap-around in spatial domain.
    4.  **Sampling (Resolution Loss):** Performs FFT cropping. This is the physically correct way to
        simulate "Low Resolution." MRI resolution is defined by how far out in K-space we sample (k-max).
        Cropping the high frequencies in K-space is exactly what happens when a scanner acquires a lower matrix size.
    5.  **Thermal Noise:** Adds Rician/Gaussian noise to simulate electronic noise in the receive coils.
    """

    def __init__(
        self,
        volume_res: List[float],
        target_res: List[float],
        output_shape: List[int],
        prob_motion: float = 0.5,
        prob_spike: float = 0.5,
        prob_aliasing: float = 0.02,
        prob_noise: float = 0.95,
        noise_std: float = 0.05,
        motion_intensity: float = 1.5,
        spike_intensity: float = 0.04,
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
        self.physics_engine = SliceProfilePhysics(profile_type='trapezoid', edge_width=0.1)

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

            # --- STEP 1: Slice Profile Blurring ---
            img = self.physics_engine(
                img, 
                resolution=self.volume_res.to(device), # Current HR resolution
                thickness=thk # Target slice thickness
            )


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
        apply_intensity_aug: If True, apply intensity augmentation to LR
        clip_to_unit_range: If True, clip outputs to [0, 1] range
    """

    def __init__(
        self,
        atlas_res: list = [1.0, 1.0, 1.0],
        target_res: list = [1.0, 1.0, 1.0],
        output_shape: list = [128, 128, 128],
        # Probabilities
        prob_motion: float = 0.5,
        prob_spike: float = 0.5,
        prob_aliasing: float = 0.02,
        prob_bias_field: float = 0.5,
        prob_noise: float = 0.95,
        prob_gamma: float = 0.5,
        # Resolution simulation
        min_resolution: list = [1.0, 1.0, 1.0],
        max_res_aniso: list = [9.0, 9.0, 9.0],
        randomise_res: bool = True,
        # Toggles
        apply_intensity_aug: bool = True,
        clip_to_unit_range: bool = True,
        # Augmentation
        gamma_std: float = 0.5,
        channel_wise: bool = False,
        noise_std: float = 0.05,
        motion_intensity: float = 1.5,
        spike_intensity: float = 0.04,
    ):
        self.atlas_res = atlas_res
        self.target_res = target_res
        self.output_shape = output_shape
        self.prob_bias_field = prob_bias_field
        self.apply_intensity_aug = apply_intensity_aug
        self.clip_to_unit_range = clip_to_unit_range
        self.randomise_res = randomise_res
        
        # 1. Resolution Sampler
        if randomise_res:
            self.res_sampler = SampleResolution(
                min_resolution=min_resolution,
                max_res_iso=None,
                max_res_aniso=max_res_aniso,
                prob_iso=0.0, 
                prob_min=0.05,
                return_thickness=True,
            )

        # 2. Bias Field
        self.bias = BiasFieldCorruption(
            bias_field_std=0.3, bias_scale=0.025, prob=1.0 
        )

        # 3. Intensity Augmentation
        if apply_intensity_aug:
            # Note: Gamma is applied to normalized data, which works fine for [0,1]
            self.intensity_aug = IntensityAugmentation(
                clip=False, # Don't clip here, we handle it globally
                gamma_std=gamma_std, 
                channel_wise=channel_wise,
                prob_gamma=prob_gamma,
            )

        # 4. MRI Physics Simulator
        self.artifact_simulator = MRIArtifactSimulator(
            volume_res=atlas_res,
            target_res=target_res,
            output_shape=output_shape,
            prob_motion=prob_motion,
            prob_spike=prob_spike,
            prob_aliasing=prob_aliasing,
            prob_noise=prob_noise,
            noise_std=noise_std,
            spike_intensity=spike_intensity,
            motion_intensity=motion_intensity,
        )

        # Normalization helper
        self.normalizer = ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
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
        Generate paired LR-HR training data with Consistent Intensity.
        """
        batch_size = hr_images.shape[0]
        device = hr_images.device

        # === STEP 1: NORMALIZE HR TO FIXED TARGET DOMAIN ===
        # Normalize the HR image to [0, 1] immediately.
        # This establishes the "Ground Truth" intensity space.
        hr_augmented = self._normalize_image(hr_images)

        # === STEP 2: CREATE LR FROM NORMALIZED HR ===
        # Clone the normalized HR. Any degradation applied now is 
        # relative to this 0-1 scale.
        lr_images = hr_augmented.clone()

        # === STEP 3: APPLY DEGRADATIONS ===
        
        # A. Bias Field (Multiplicative shading)
        # This will shift pixel values locally (e.g., 0.8 -> 0.6 or 0.8 -> 0.95)
        # We WANT this mismatch so the network learns to correct it.
        if torch.rand(1).item() < self.prob_bias_field:
            lr_images = self.bias(lr_images)

        # B. Intensity Augmentation (Gamma)
        # Simulates different contrast settings (e.g., T1 vs T2-like contrast shifts)
        if self.apply_intensity_aug:
            lr_images = self.intensity_aug(lr_images)

        # C. Physics Simulation (PSF, downsampling, noise, motion, aliasing)
        resolution = None
        thickness = None

        if self.randomise_res:
            resolution, thickness = self.res_sampler(batch_size)
            resolution = resolution.to(device)
            thickness = thickness.to(device)
            lr_images = self.artifact_simulator(lr_images, resolution, thickness)
        else:
            resolution = torch.tensor([self.atlas_res] * batch_size, dtype=torch.float32, device=device)
            thickness = resolution.clone()
            lr_images = self.artifact_simulator(lr_images, resolution, thickness)

        # === STEP 3: REALISTIC LR INTENSITY NORMALIZATION ===

        if self.clip_to_unit_range:
            # Per-volume soft clipping + min-max normalization to [0, 1]
            lr_norm = []
            for b in range(batch_size):
                lr_b = lr_images[b:b+1]

                # (1) Compute soft clipping bounds
                low = torch.quantile(lr_b, 0.005)
                high = torch.quantile(lr_b, 0.995)

                # (2) Apply clipping
                lr_b = torch.clamp(lr_b, low, high)

                # (3) Global min-max normalization
                min_val = lr_b.min()
                max_val = lr_b.max()
                lr_b = (lr_b - min_val) / (max_val - min_val + 1e-8)

                lr_norm.append(lr_b)

            lr_images = torch.cat(lr_norm, dim=0)

            # HR is already normalized by percentiles; clip tiny float drift
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
    if is_training:
        transforms.append(
            RandAffined(
                keys=["image"],
                prob=0.3,                       
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                shear_range=None,
                translate_range=(5, 5, 5),     # optional
                mode="bilinear",
                padding_mode="border",
            )
        )
    transforms.append(ToTensord(keys=["image"]))
    transform = Compose(transforms)

    if use_cache:
        dataset = CacheDataset(data=data_dicts, transform=transform, cache_rate=1.0, num_workers=4)
    else:
        dataset = Dataset(data=data_dicts, transform=transform)

    return GeneratorDataset(dataset, generator, return_resolution)