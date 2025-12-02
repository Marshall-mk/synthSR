"""
PyTorch Inference Script

Usage:
    python test.py \
        --input /path/to/input.nii.gz \
        --output scan_synthsr.nii.gz \
        --model ./models/regression_unet_epoch_005.pth \
        --atlas_res 1.0 1.0 1.0

License: Apache 2.0
"""

import os
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
)

from src.utils import (
    pad_to_multiple_of_32,
    unpad_volume,
    load_model_from_checkpoint,
)


def create_inference_transforms(atlas_res=[1.0, 1.0, 1.0]):
    """
    Create MONAI preprocessing transforms for inference.

    Args:
        atlas_res: Target resolution [x, y, z] in mm

    Returns:
        MONAI Compose transform
    """
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=False),  # Keep metadata for saving
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),  # Add channel dimension if not present
            Orientationd(keys=["image"], axcodes="RAS", labels=None),  # Align to RAS orientation
            Spacingd(keys=["image"], pixdim=atlas_res, mode="bilinear"),  # Resample to target resolution
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0, upper=100, b_min=0.0, b_max=1.0, clip=True
            ),  # Normalize to [0, 1]
        ]
    )


def predict_single_volume(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    atlas_res: list = [1.0, 1.0, 1.0],
    device: str = "cuda",
    use_tta: bool = True,
    save_intermediates: bool = False,
):
    """
    Run inference on a single volume using MONAI transforms.

    Args:
        model: Trained UNet model
        input_path: Path to input NIfTI file
        output_path: Path to save output
        atlas_res: Target resolution in mm [x, y, z]
        device: 'cuda' or 'cpu'
        use_tta: Use test-time augmentation (flipping)
        save_intermediates: Save intermediate volumes for debugging
    """
    print(f"Processing: {input_path}")

    # Create MONAI preprocessing pipeline
    transforms = create_inference_transforms(atlas_res)

    # Load and preprocess with MONAI
    print("  Loading and preprocessing with MONAI transforms...")
    data = transforms({"image": input_path})

    # Extract volume and metadata
    if isinstance(data, dict):
        volume = data["image"]
        meta = data.get("image_meta_dict", {})
    else:
        volume = data
        meta = {}

    # Convert to numpy if tensor
    if isinstance(volume, torch.Tensor):
        volume_np = volume.cpu().numpy()
    else:
        volume_np = np.array(volume)

    # Remove channel dimension for shape reporting (C, D, H, W) -> (D, H, W)
    if volume_np.ndim == 4 and volume_np.shape[0] == 1:
        volume_for_info = volume_np[0]
    else:
        volume_for_info = volume_np

    print(f"  After MONAI preprocessing: {volume_for_info.shape}")

    # Get affine from metadata if available
    affine = None
    if "affine" in meta:
        affine = meta["affine"]
        if isinstance(affine, torch.Tensor):
            affine = affine.cpu().numpy()
        print("  Using affine from MONAI metadata.")

    # If affine still not found, try loading directly from original file
    if affine is None:
        try:
            print("  Affine not in metadata, loading directly from original image...")
            original_img = nib.load(input_path)
            affine = original_img.affine.copy()

            # Adjust affine for the resampled resolution
            # The Spacingd transform updates the affine, but if metadata is lost, we need to scale it
            original_pixdim = original_img.header.get_zooms()[:3]
            scale_factors = np.array(original_pixdim) / np.array(atlas_res)

            # Update the affine to reflect the new resolution
            affine[:3, :3] = affine[:3, :3] / scale_factors[:, np.newaxis] * scale_factors
            print(f"  Successfully loaded affine from original image (origin: {affine[:3, 3]})")
        except Exception as e:
            print(f"  Warning: Could not load affine from original file ({e})")
            print("  Using fallback affine (identity with atlas resolution).")
            # Fallback: create identity affine with atlas resolution
            affine = np.diag([atlas_res[0], atlas_res[1], atlas_res[2], 1.0])

    # Save intermediate preprocessed volume if requested
    if save_intermediates:
        temp_nii = nib.Nifti1Image(volume_for_info, affine)
        nib.save(temp_nii, "temp_preprocessed.nii.gz")
        print("  Saved intermediate: temp_preprocessed.nii.gz")

    # Step 4: Pad to multiples of 32 (required for UNet with 5 levels)
    volume_padded, pad_before, original_shape = pad_to_multiple_of_32(volume_for_info)
    print(f"  After padding to multiple of 32: {volume_padded.shape}")

    # Save intermediate padded volume if requested
    if save_intermediates:
        temp_nii = nib.Nifti1Image(volume_padded, affine)
        nib.save(temp_nii, "temp_padded.nii.gz")
        print("  Saved intermediate: temp_padded.nii.gz")

    # Step 5: Convert to tensor (add batch and channel dims)
    input_tensor = (
        torch.from_numpy(volume_padded).float().unsqueeze(0).unsqueeze(0)
    )  # (1, 1, D, H, W)
    input_tensor = input_tensor.to(device)

    # Step 6: Run inference
    model.eval()
    with torch.no_grad():
        if use_tta:
            # Test-time augmentation: average predictions from original and flipped
            print("  Running inference with TTA...")
            pred1 = model(input_tensor)
            pred2 = torch.flip(model(torch.flip(input_tensor, dims=[2])), dims=[2])
            output = 0.5 * (pred1 + pred2)
        else:
            print("  Running inference...")
            output = model(input_tensor)

    # Step 7: Convert back to numpy and remove batch/channel dims
    output = output.squeeze(0).squeeze(0).cpu().numpy()

    # Save intermediate output volume if requested
    if save_intermediates:
        temp_nii = nib.Nifti1Image(output, affine)
        nib.save(temp_nii, "temp_output.nii.gz")
        print("  Saved intermediate: temp_output.nii.gz")

    # Step 8: Unpad to original shape
    output = unpad_volume(output, pad_before, original_shape)
    print(f"  After unpadding: {output.shape}")

    # Step 9: Post-process (scale to 0-255 range, clip)
    output = 255.0 * output
    output = np.clip(output, 0, 255)

    # Step 10: Save with affine matrix
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_nii = nib.Nifti1Image(output, affine)
    nib.save(out_nii, output_path)
    print(f"  Saved to: {output_path}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")


def predict_batch(
    input_paths: list,
    output_paths: list,
    model_path: str,
    atlas_res: list = [1.0, 1.0, 1.0],
    device: str = "cuda",
    use_tta: bool = True,
    save_intermediates: bool = False,
):
    """Process multiple volumes in batch."""
    print("=" * 80)
    print("SynthSR Inference - PyTorch with MONAI")
    print("=" * 80)

    # Load model with automatic architecture reconstruction
    print(f"\nLoading model from {model_path}...")
    model, _ = load_model_from_checkpoint(model_path, device=device)
    print(f"Model loaded successfully!")
    print(f"Device: {device}")
    print(f"Atlas resolution: {atlas_res} mm")
    print(f"Test-time augmentation: {use_tta}")
    print(f"Processing {len(input_paths)} volumes...\n")

    # Process each volume
    for idx, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        print(f"[{idx + 1}/{len(input_paths)}]")
        try:
            predict_single_volume(
                model=model,
                input_path=input_path,
                output_path=output_path,
                atlas_res=atlas_res,
                device=device,
                use_tta=use_tta,
                save_intermediates=save_intermediates,
            )
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            continue
        print()

    print("=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynthSR Inference with MONAI")

    # Input/output arguments
    parser.add_argument(
        "--input", type=str, required=True, help="Input image file or directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output image file or directory"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )

    # Preprocessing arguments
    parser.add_argument(
        "--atlas_res",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Atlas resolution in mm (e.g., --atlas_res 1.0 1.0 1.0)",
    )

    # Inference arguments
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--tta", action="store_true", help="Use test-time augmentation (flipping)"
    )
    parser.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Disable test-time augmentation",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate volumes for debugging",
    )
    parser.set_defaults(tta=True)

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Prepare input/output paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single file
        input_paths = [str(input_path)]
        output_paths = [str(output_path)]
    elif input_path.is_dir():
        # Directory
        input_paths = sorted(
            [str(p) for p in input_path.glob("*.nii.gz")]
            + [str(p) for p in input_path.glob("*.nii")]
        )

        if len(input_paths) == 0:
            raise ValueError(f"No .nii or .nii.gz files found in {input_path}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output paths
        output_paths = [
            str(output_path / (Path(ip).stem.replace(".nii", "") + "_synthsr.nii.gz"))
            for ip in input_paths
        ]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    # Run inference
    predict_batch(
        input_paths=input_paths,
        output_paths=output_paths,
        model_path=args.model,
        atlas_res=args.atlas_res,
        device=args.device,
        use_tta=args.tta,
        save_intermediates=args.save_intermediates,
    )
