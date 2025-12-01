"""
Inference script for diffusion-based super-resolution model.

This script loads a trained diffusion model and generates high-resolution
images from low-resolution inputs.
"""

import os
import argparse
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.diff_models import DiffusionSuperResolution, create_diffusion_model
from src.utils import pad_to_multiple_of_32, unpad_volume

from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    Compose,
)


def load_diffusion_model(
    checkpoint_path: str,
    device: str = "cuda",
    image_size: tuple = (128, 128, 128),
    model_size: str = "base",
) -> DiffusionSuperResolution:
    """
    Load diffusion model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        image_size: Image size used during training (D, H, W)
        model_size: Model size preset ('tiny', 'small', 'base', 'large')

    Returns:
        Loaded diffusion model
    """
    print(f"Loading diffusion model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    scheduler_type = model_config.get("scheduler_type", "ddpm")
    num_train_timesteps = model_config.get("num_train_timesteps", 1000)
    beta_schedule = model_config.get("beta_schedule", "linear")
    output_shape = model_config.get("output_shape", image_size)

    # Try to get model_size from config, otherwise use default
    if "model_size" in model_config:
        model_size = model_config["model_size"]

    print(f"Model config: scheduler={scheduler_type}, timesteps={num_train_timesteps}")
    print(f"Model size: {model_size}, output_shape: {output_shape}")

    # Create model using the factory function
    model = create_diffusion_model(
        image_size=output_shape,
        in_channels=1,
        model_size=model_size,
        scheduler_type=scheduler_type,
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
    )

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    print(f"Using NEW hybrid conditioning: LR concatenation + resolution cross-attention")
    return model


def preprocess_image(
    input_path: str,
    target_spacing: list = None,
) -> tuple:
    """
    Load and preprocess input image.

    Args:
        input_path: Path to input NIfTI file
        target_spacing: Optional target voxel spacing [x, y, z] in mm

    Returns:
        Tuple of (preprocessed_volume, metadata, original_spacing)
    """
    print(f"Loading input image: {input_path}")

    # Build preprocessing pipeline
    transforms = [
        LoadImage(image_only=False),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
    ]

    if target_spacing is not None:
        transforms.append(Spacing(pixdim=target_spacing, mode="bilinear"))

    transform = Compose(transforms)

    # Load and preprocess
    img_obj = transform(input_path)
    volume = img_obj[0]  # Get tensor
    metadata = img_obj[1]  # Get metadata

    # Get original spacing
    original_spacing = metadata.get("pixdim", [1.0, 1.0, 1.0])[:3]

    # Ensure shape is (C, D, H, W)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)

    print(f"Input volume shape: {volume.shape}")
    print(f"Original spacing: {original_spacing}")

    # Normalize to [0, 1]
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    return volume, metadata, original_spacing


def infer_diffusion(
    model: DiffusionSuperResolution,
    lr_image: torch.Tensor,
    resolution: torch.Tensor = None,
    num_inference_steps: int = 50,
    device: str = "cuda",
    use_padding: bool = True,
) -> torch.Tensor:
    """
    Run diffusion inference on low-resolution image.

    Args:
        model: Trained diffusion model
        lr_image: Low-resolution input (C, D, H, W)
        resolution: Resolution parameters (3,)
        num_inference_steps: Number of denoising steps
        device: Device to run on
        use_padding: Whether to pad input to multiple of 32

    Returns:
        Super-resolved image (C, D, H, W)
    """
    model.eval()

    # Pad if needed
    if use_padding:
        original_shape = lr_image.shape[1:]  # (D, H, W)
        padded_volume, pad_before, orig_shape = pad_to_multiple_of_32(
            lr_image.squeeze(0).numpy()
        )
        lr_image = torch.from_numpy(padded_volume).unsqueeze(0).float()
        print(f"Padded input shape: {lr_image.shape}")

    # Move to device
    lr_image = lr_image.unsqueeze(0).to(device)  # Add batch dimension

    # Prepare resolution conditioning
    if resolution is not None:
        resolution = resolution.unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    print(f"Running diffusion inference with {num_inference_steps} steps...")
    with torch.no_grad():
        sr_image = model.sample(
            lr_condition=lr_image,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
        )

    # Remove batch dimension
    sr_image = sr_image.squeeze(0).cpu()

    # Remove padding if applied
    if use_padding:
        sr_image = sr_image.squeeze(0).numpy()  # Remove channel dim
        sr_image = unpad_volume(sr_image, pad_before, orig_shape)
        sr_image = torch.from_numpy(sr_image).unsqueeze(0)  # Add channel dim back

    return sr_image


def save_result(
    output_path: str,
    sr_image: torch.Tensor,
    metadata: dict,
):
    """
    Save super-resolved image to NIfTI file.

    Args:
        output_path: Path to save output
        sr_image: Super-resolved image (C, D, H, W)
        metadata: Metadata from original image
    """
    # Convert to numpy
    sr_image = sr_image.squeeze(0).numpy()  # Remove channel dim

    print(f"Output image shape: {sr_image.shape}")

    # Get affine from metadata
    affine = metadata.get("affine", np.eye(4))

    # Create NIfTI image
    output_nii = nib.Nifti1Image(sr_image, affine=affine)

    # Copy header metadata if available
    if "original_affine" in metadata:
        output_nii.header["pixdim"] = metadata.get(
            "pixdim", output_nii.header["pixdim"]
        )

    # Save
    nib.save(output_nii, output_path)
    print(f"Saved super-resolved image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diffusion-based super-resolution inference"
    )

    # Input/output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input low-resolution NIfTI file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output super-resolved NIfTI file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained diffusion model checkpoint",
    )

    # Inference parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=None,
        help="Target voxel spacing [x, y, z] in mm (optional)",
    )
    parser.add_argument(
        "--input_resolution",
        type=float,
        nargs=3,
        default=None,
        help="Input resolution for conditioning [x, y, z] in mm (optional)",
    )
    parser.add_argument(
        "--no_padding",
        action="store_true",
        help="Disable padding to multiple of 32",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Image size used during training [D, H, W] (default: 128 128 128)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        choices=["tiny", "small", "base", "large"],
        help="Model size preset (default: base)",
    )

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    model = load_diffusion_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=tuple(args.image_size),
        model_size=args.model_size,
    )

    # Preprocess input
    lr_image, metadata, original_spacing = preprocess_image(
        args.input,
        target_spacing=args.target_spacing,
    )

    # Prepare resolution conditioning
    resolution = None
    if args.input_resolution is not None:
        resolution = torch.tensor(args.input_resolution, dtype=torch.float32)
        print(f"Using resolution conditioning: {args.input_resolution}")
    else:
        # Use original spacing as resolution
        resolution = torch.tensor(original_spacing, dtype=torch.float32)
        print(f"Using original spacing as resolution: {original_spacing}")

    # Run inference
    sr_image = infer_diffusion(
        model=model,
        lr_image=lr_image,
        resolution=resolution,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        use_padding=not args.no_padding,
    )

    # Save result
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_result(args.output, sr_image, metadata)

    print("Inference complete!")


if __name__ == "__main__":
    main()
