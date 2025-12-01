import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import json
import datetime
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List, Union

from monai.losses import SSIMLoss as MonaiSSIM

from .models import CustomUNet3D


# =============================================================================
# Padding Utilities for Inference
# =============================================================================


def pad_to_multiple_of_32(
    volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad volume to make all dimensions multiples of 32 (centered padding).

    This is used for inference with UNet models that require input dimensions
    to be divisible by 2^n_levels (e.g., 32 for 5 levels).

    Args:
        volume: Input volume array (D, H, W)

    Returns:
        Tuple of (padded volume, padding before, original shape)

    Example:
        >>> volume = np.random.rand(100, 120, 90)
        >>> padded, pad_before, orig_shape = pad_to_multiple_of_32(volume)
        >>> print(padded.shape)  # (128, 128, 96) - next multiples of 32
        >>> # Later unpad: volume = padded[pad_before[0]:pad_before[0]+orig_shape[0], ...]
    """
    shape = np.array(volume.shape)
    # Calculate target shape (next multiple of 32)
    target_shape = (np.ceil(shape / 32.0) * 32).astype("int")

    # Calculate padding (centered)
    padding = target_shape - shape
    pad_before = np.floor(padding / 2).astype("int")
    pad_after = padding - pad_before

    # Pad the volume
    padded = np.pad(
        volume,
        [(pad_before[i], pad_after[i]) for i in range(3)],
        mode="constant",
        constant_values=0,
    )

    return padded, pad_before, shape


def unpad_volume(
    volume: np.ndarray, pad_before: np.ndarray, original_shape: np.ndarray
) -> np.ndarray:
    """
    Remove padding from volume that was added by pad_to_multiple_of_32.

    Args:
        volume: Padded volume array
        pad_before: Padding amounts before (from pad_to_multiple_of_32)
        original_shape: Original shape before padding (from pad_to_multiple_of_32)

    Returns:
        Unpadded volume with original shape

    Example:
        >>> padded, pad_before, orig_shape = pad_to_multiple_of_32(volume)
        >>> # ... process padded volume ...
        >>> result = unpad_volume(processed, pad_before, orig_shape)
    """
    return volume[
        pad_before[0] : pad_before[0] + original_shape[0],
        pad_before[1] : pad_before[1] + original_shape[1],
        pad_before[2] : pad_before[2] + original_shape[2],
    ]


# =============================================================================
# Data Loading Utilities
# =============================================================================


def load_image_paths_from_csv(
    csv_path: Union[str, Path],
    base_dir: Union[str, Path],
    split: str = "train",
    acquisition_types: Optional[List[str]] = None,
    mri_classifications: Optional[List[str]] = None,
    filter_4d: bool = True,
    log_filtered_path: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Load image paths from a CSV file with filtering by split, acquisition type, and MRI classification.

    Args:
        csv_path: Path to CSV file
        base_dir: Base directory to prepend to relative paths
        split: Data split to filter for ('train', 'val', or 'test')
        acquisition_types: List of acquisition types to include (default: ['3D'] only)
                          Set to None to include all types
        mri_classifications: List of MRI classifications to include (e.g., ['T1', 'T2', 'FLAIR'])
                            Set to None to include all classifications
        filter_4d: If True, filters out 4D images (with time dimension) using 'dimensions' column
        log_filtered_path: Optional path to save list of filtered 4D images (CSV format)

    Returns:
        List of absolute image paths

    CSV Format:
        The CSV should have the following columns:
        - relative_path: Relative path to the image file
        - mr_acquisition_type: Type of MR acquisition ('3D' or '2D')
        - split: Data split ('train', 'val', or 'test')
        - MRI_classification (optional): MRI classification type ('T1', 'T2', 'FLAIR', etc.)
        - dimensions (optional): Image dimensions as tuple string, e.g., "(256, 256, 128)"

    Example CSV:
        relative_path,mr_acquisition_type,split,MRI_classification,dimensions
        images/scan001.nii.gz,3D,train,T1,"(256, 256, 128)"
        images/scan002.nii.gz,2D,train,T2,"(256, 256, 128)"
        images/scan003.nii.gz,3D,val,FLAIR,"(256, 256, 128, 10)"
        images/scan004.nii.gz,3D,test,T1,"(256, 256, 128)"

    Example:
        >>> # Load 3D training images only (filtering out 4D) with logging
        >>> train_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='train',
        ...     filter_4d=True,
        ...     log_filtered_path='./model/filtered_4d_train.csv'
        ... )
        >>>
        >>> # Load T1 and T2 training images only
        >>> train_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='train',
        ...     mri_classifications=['T1', 'T2']
        ... )
        >>>
        >>> # Load validation images of all types (including 4D)
        >>> val_paths = load_image_paths_from_csv(
        ...     'data.csv',
        ...     base_dir='/data/mri',
        ...     split='val',
        ...     acquisition_types=None,
        ...     filter_4d=False
        ... )
    """
    csv_path = Path(csv_path)
    base_dir = Path(base_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_columns = ["relative_path", "mr_acquisition_type", "split"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    # Filter by split
    df_filtered = df[df["split"] == split].copy()

    if len(df_filtered) == 0:
        raise ValueError(
            f"No images found for split '{split}'. "
            f"Available splits: {df['split'].unique().tolist()}"
        )

    # Filter by acquisition type if specified (default to 3D only)
    if acquisition_types is None:
        acquisition_types = ["3D"]

    df_filtered = df_filtered[
        df_filtered["mr_acquisition_type"].isin(acquisition_types)
    ]

    if len(df_filtered) == 0:
        raise ValueError(
            f"No images found for split '{split}' with acquisition types {acquisition_types}. "
            f"Available acquisition types: {df[df['split'] == split]['mr_acquisition_type'].unique().tolist()}"
        )

    # Filter by MRI classification if specified
    if mri_classifications is not None and "MRI_classification" in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered["MRI_classification"].isin(mri_classifications)
        ]

        if len(df_filtered) == 0:
            raise ValueError(
                f"No images found for split '{split}' with MRI classifications {mri_classifications}. "
                f"Available MRI classifications: {df[df['split'] == split]['MRI_classification'].unique().tolist()}"
            )

        print(f"Filtered by MRI classifications: {mri_classifications}")
    elif mri_classifications is not None and "MRI_classification" not in df_filtered.columns:
        print(
            "Warning: 'MRI_classification' column not found in CSV. "
            "Cannot filter by MRI classification. Continuing with all MRI types."
        )

    # Filter out 4D images if requested
    if filter_4d and "dimensions" in df_filtered.columns:
        initial_count = len(df_filtered)

        def is_3d_image(dim_str):
            """Check if dimensions string represents a 3D image (not 4D)."""
            if pd.isna(dim_str):
                # If dimensions column is missing for this row, assume it's okay (will be caught later)
                return True

            try:
                # Parse dimension string - could be "(256, 256, 128)" or similar
                dim_str = str(dim_str).strip()
                # Remove parentheses and split by comma
                dim_str = dim_str.replace("(", "").replace(")", "").strip()
                dims = [int(d.strip()) for d in dim_str.split(",") if d.strip()]
                # Return True if exactly 3 dimensions (3D image)
                return len(dims) == 3
            except Exception as e:
                # If parsing fails, log warning and assume it's okay
                print(f"Warning: Could not parse dimensions '{dim_str}': {e}")
                return True

        # Identify 4D images before filtering
        is_3d_mask = df_filtered["dimensions"].apply(is_3d_image)
        filtered_4d_images = df_filtered[~is_3d_mask].copy()

        # Apply filter
        df_filtered = df_filtered[is_3d_mask].copy()

        filtered_count = len(filtered_4d_images)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} 4D images (with time dimension)")

            # Save filtered images to log file if requested
            if log_filtered_path is not None:
                log_path = Path(log_filtered_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                # Add reason column
                filtered_4d_images["filter_reason"] = "4D image (time dimension)"

                # Save to CSV
                filtered_4d_images.to_csv(log_path, index=False)
                print(f"Saved list of filtered 4D images to: {log_path}")

        if len(df_filtered) == 0:
            raise ValueError(
                f"No 3D images found for split '{split}' after filtering out 4D images. "
                f"All {initial_count} images had 4 dimensions."
            )
    elif filter_4d and "dimensions" not in df_filtered.columns:
        print(
            "Warning: 'dimensions' column not found in CSV. Cannot filter 4D images. "
            "Consider adding a 'dimensions' column to your CSV for better filtering."
        )

    # Convert relative paths to absolute paths
    image_paths = []
    for rel_path in df_filtered["relative_path"]:
        abs_path = base_dir / rel_path
        if not abs_path.exists():
            print(f"Warning: File not found: {abs_path}")
        else:
            image_paths.append(abs_path)

    if len(image_paths) == 0:
        raise ValueError(
            f"No valid image files found for split '{split}'. "
            f"Check that files exist in {base_dir}"
        )

    # Build summary message
    summary = f"Loaded {len(image_paths)} {split} images (acquisition types: {acquisition_types}"
    if mri_classifications is not None:
        summary += f", MRI classifications: {mri_classifications}"
    summary += ")"
    print(summary)

    return image_paths


def get_image_paths(
    image_dir=None,
    csv_file=None,
    base_dir=None,
    split="train",
    model_dir=None,
    mri_classifications=None,
):
    """
    Get image paths from either directory or CSV file.

    Args:
        image_dir: Directory containing images (mutually exclusive with csv_file)
        csv_file: CSV file with image metadata (mutually exclusive with image_dir)
        base_dir: Base directory for relative paths in CSV (required if csv_file is provided)
        split: Data split for CSV ('train', 'val', or 'test')
        model_dir: Model directory for saving filtered images log (optional)
        mri_classifications: List of MRI classifications to include (e.g., ['T1', 'T2', 'FLAIR'])
                           Only applicable when using csv_file

    Returns:
        List of image paths
    """
    if csv_file is not None:
        if base_dir is None:
            raise ValueError("--base_dir is required when using --csv_file")

        # Set up log path for filtered 4D images if model_dir is provided
        log_filtered_path = None
        if model_dir is not None:
            log_filtered_path = os.path.join(
                model_dir, f"filtered_4d_images_{split}.csv"
            )

        return load_image_paths_from_csv(
            csv_file,
            base_dir,
            split=split,
            mri_classifications=mri_classifications,
            filter_4d=True,  # Filter out 4D images by default
            log_filtered_path=log_filtered_path,
        )
    elif image_dir is not None:
        # Get all .nii.gz files from directory
        image_dir = Path(image_dir)
        return sorted([str(p) for p in image_dir.glob("*.nii.gz")])
    else:
        raise ValueError("Either --hr_image_dir or --csv_file must be provided")


# =============================================================================
# Model Checkpoint Management
# =============================================================================


def save_model_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = None,
    val_loss: float = None,
    model_type: str = "unet3d",
    model_config: Optional[Dict[str, Any]] = None,
    **extra_data,
):
    """
    Save model checkpoint with complete architecture configuration.

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        loss: Training loss
        val_loss: Validation loss (optional)
        model_type: Type of model ("unet3d", "parameter_aware_unet", etc.)
        model_config: Model architecture configuration dict
        **extra_data: Additional data to save in checkpoint
    """
    checkpoint = {
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if loss is not None:
        checkpoint["loss"] = loss

    if val_loss is not None:
        checkpoint["val_loss"] = val_loss

    # Save model architecture configuration
    if model_config is not None:
        checkpoint["model_config"] = model_config

    # Add any extra data
    checkpoint.update(extra_data)

    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, filepath)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    strict: bool = True,
) -> tuple[nn.Module, Dict[str, Any]]:
    """
    Load any model (CustomUNet3D or MONAI) from checkpoint with automatic architecture reconstruction.

    This function reads the architecture configuration from the checkpoint
    and reconstructs the appropriate model type.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
        strict: Whether to strictly enforce state_dict key matching

    Returns:
        model: Loaded model (CustomUNet3D or MONAI model)
        checkpoint: Full checkpoint dict with metadata
    """
    from .models import create_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Check if model config is saved in checkpoint
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
        model_arch = config.get("model_architecture", "unet3d")

        if model_arch == "unet3d":
            # Load custom UNet3D
            model = CustomUNet3D(**config)
        elif model_arch == "monai":
            # Load MONAI model using create_model
            model_name = config["model_name"]
            img_size = config.get("output_shape", (128, 128, 128))
            in_channels = config.get("in_channels", 1)
            out_channels = config.get("out_channels", 1)

            model = create_model(
                model_name=model_name,
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                device=None,
            )
        else:
            raise ValueError(f"Unknown model architecture: {model_arch}")
    else:
        # Fallback: Try to infer from saved parameters or use defaults
        # This handles old checkpoints that don't have model_config
        print(
            "Warning: model_config not found in checkpoint. Using default configuration."
        )
        print(
            "For best results, re-save checkpoints with model_utils.save_model_checkpoint()"
        )

        model = CustomUNet3D(
            nb_features=checkpoint.get("nb_features", 24),
            input_shape=(1, 128, 128, 128),  # Default shape
            nb_levels=checkpoint.get("nb_levels", 5),
            conv_size=checkpoint.get("conv_size", 3),
            nb_labels=checkpoint.get("nb_labels", 1),
            feat_mult=checkpoint.get("feat_mult", 2),
            final_pred_activation=checkpoint.get("final_pred_activation", "linear"),
            nb_conv_per_level=checkpoint.get("nb_conv_per_level", 2),
        )

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif "regression_unet_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["regression_unet_state_dict"], strict=strict)
    else:
        # Assume the entire checkpoint is the state dict
        model.load_state_dict(checkpoint, strict=strict)

    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_unet3d_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    strict: bool = True,
) -> tuple[nn.Module, Dict[str, Any]]:
    """
    Backward compatibility wrapper for load_model_from_checkpoint.

    Deprecated: Use load_model_from_checkpoint instead.
    This function is kept for backward compatibility with existing code.
    """
    return load_model_from_checkpoint(checkpoint_path, device, strict)


def find_latest_checkpoint(model_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint in the model directory.

    Searches for checkpoint files matching the pattern 'regression_unet_epoch_*.pth'
    and returns the one with the highest epoch number. This enables automatic
    training resumption without manually specifying checkpoint paths.

    Args:
        model_dir: Directory containing checkpoint files

    Returns:
        Path to the most recent checkpoint, or None if no checkpoints found

    Example:
        >>> checkpoint = find_latest_checkpoint("./models")
        >>> if checkpoint:
        ...     print(f"Found checkpoint: {checkpoint}")
        ...     # Output: Found checkpoint: ./models/regression_unet_epoch_050.pth
        ... else:
        ...     print("No checkpoints found, starting from scratch")
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None

    # Find all checkpoint files matching the pattern
    checkpoint_pattern = re.compile(r"regression_unet_epoch_(\d+)\.pth")
    checkpoints = []

    for file_path in model_dir.glob("regression_unet_epoch_*.pth"):
        match = checkpoint_pattern.match(file_path.name)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, str(file_path)))

    if not checkpoints:
        return None

    # Return checkpoint with highest epoch number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint = checkpoints[0][1]

    return latest_checkpoint


# =============================================================================
# Training Configuration Management
# =============================================================================


def save_training_config(
    model_dir: str,
    args,  # argparse.Namespace or similar object with attributes
    n_train_samples: int,
    n_val_samples: int,
    training_stage: str = "stage1",
):
    """
    Save training configuration and metadata to JSON file.

    Args:
        model_dir: Directory where model is saved
        args: Command-line arguments namespace (or object with similar attributes)
        n_train_samples: Number of training samples
        n_val_samples: Number of validation samples
        training_stage: "stage1" or "stage2"
    """
    config = {
        "training_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "training_stage": training_stage,
            "n_train_samples": n_train_samples,
            "n_val_samples": n_val_samples,
            "resumed_from_checkpoint": getattr(args, "checkpoint", None),
        },
        "model_parameters": {
            "model_architecture": getattr(args, "model_architecture", None),
            "model_name": getattr(args, "model_name", None),
            "nb_features": getattr(args, "nb_features", 24),
            "nb_levels": getattr(args, "nb_levels", 5),
        },
        "training_parameters": {
            "epochs": getattr(args, "epochs", None),
            "batch_size": getattr(args, "batch_size", None),
            "learning_rate": getattr(args, "learning_rate", None),
            "save_interval": getattr(args, "save_interval", None),
            "device": getattr(args, "device", None),
            "use_cache": getattr(args, "use_cache", None),
        },
        "data_parameters": {
            "output_shape": getattr(args, "output_shape", None),
            "atlas_res": getattr(args, "atlas_res", None),
            "target_res": [1.0, 1.0, 1.0],
            "min_resolution": getattr(args, "min_resolution", None),
            "max_res_aniso": getattr(args, "max_res_aniso", None),
            "auto_detect_res": getattr(args, "auto_detect_res", None),
        },
        "augmentation_parameters": {
            "randomise_res": not getattr(args, "no_randomise_res", False),
            "apply_lr_deformation": not getattr(args, "no_lr_deformation", False),
            "apply_hr_deformation": not getattr(args, "no_hr_deformation", False),
            "apply_bias_field": not getattr(args, "no_bias_field", False),
            "apply_intensity_aug": not getattr(args, "no_intensity_aug", False),
            "enable_90_rotations": getattr(args, "enable_90_rotations", False),
        },
        "data_sources": {
            "hr_image_dir": getattr(args, "hr_image_dir", None),
            "val_image_dir": getattr(args, "val_image_dir", None),
            "csv_file": getattr(args, "csv_file", None),
            "base_dir": getattr(args, "base_dir", None),
        },
        "system_info": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
    }

    # Save to JSON file
    config_path = os.path.join(model_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to: {config_path}")


# =============================================================================
# Loss Functions and Metrics
# =============================================================================


class SSIMLoss(nn.Module):
    """SSIM loss for 3D images using MONAI's implementation."""

    def __init__(self):
        super().__init__()
        self.ssim = MonaiSSIM(spatial_dims=3)

    def forward(self, pred, target):
        return self.ssim(pred, target)

class L1SSIMLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.ssim_loss = MonaiSSIM(spatial_dims=3)
        self.l1 = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        ssim = self.ssim_loss(pred, target)    
        return self.alpha * l1 + (1 - self.alpha) * ssim

def get_loss_function(loss_name: str):
    """
    Get loss function by name.

    Args:
        loss_name: Name of loss function (l1, l2, huber, ssim, gaussian_nll)

    Returns:
        PyTorch loss function
    """
    if loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "l2":
        return nn.MSELoss()
    elif loss_name == "huber":
        return nn.HuberLoss(delta=1.0)
    elif loss_name == "ssim":
        return SSIMLoss()
    elif loss_name == "l1+ssim":
        return L1SSIMLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0):
    """
    Calculate evaluation metrics.

    Args:
        pred: Predicted images (B, C, D, H, W)
        target: Target images (B, C, D, H, W)
        max_val: Maximum pixel value for PSNR calculation (default: 1.0)

    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # MAE (L1)
        mae = torch.abs(pred - target).mean().item()

        # MSE (L2)
        mse = ((pred - target) ** 2).mean().item()

        # RMSE
        rmse = torch.sqrt(torch.tensor(mse)).item()

        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 10 * torch.log10(torch.tensor(max_val**2 / mse)).item()
        else:
            psnr = float("inf")

        # R² (Coefficient of Determination)
        target_mean = target.mean()
        ss_tot = ((target - target_mean) ** 2).sum().item()
        ss_res = ((target - pred) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "psnr": psnr,
            "r2": r2,
        }


# =============================================================================
# Sliding Window Inference
# =============================================================================


def sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: tuple = (128, 128, 64),
    overlap: float = 0.5,
    batch_size: int = 4,
    device: str = "cuda",
    blend_mode: str = "gaussian",
    progress: bool = True,
) -> torch.Tensor:
    """
    Perform sliding window inference on a large 3D volume using MONAI's implementation.

    Wrapper around MONAI's sliding_window_inference that divides the input volume into
    overlapping patches, runs model inference on each patch, and reconstructs the full
    prediction by blending overlapping regions. This enables inference on volumes larger
    than what fits in GPU memory during training.

    Args:
        model: PyTorch model for inference (should be in eval mode)
        volume: Input volume tensor of shape (1, C, D, H, W) or (C, D, H, W)
        patch_size: Size of patches for inference (D, H, W)
            Example: (128, 128, 64) for patches of 128×128×64
        overlap: Overlap ratio between adjacent patches [0, 1)
            - 0.0: No overlap (faster, potential edge artifacts)
            - 0.5: 50% overlap (good balance)
            - 0.75: 75% overlap (smoother, slower)
        batch_size: Number of patches to process simultaneously
            Larger = faster but more memory
        device: Device for inference ('cuda' or 'cpu')
        blend_mode: Method for blending overlapping patches
            - 'gaussian': Gaussian-weighted blending (smoother, recommended)
            - 'constant': Simple averaging
        progress: If True, show progress bar

    Returns:
        Predicted volume of same shape as input (1, C, D, H, W) or (C, D, H, W)

    Example:
        >>> from src.utils import load_unet3d_from_checkpoint, sliding_window_inference
        >>> model, _ = load_unet3d_from_checkpoint("model.pth", device="cuda")
        >>> model.eval()
        >>>
        >>> # Full-size volume (too large for training batch)
        >>> volume = torch.randn(1, 1, 320, 320, 128).cuda()
        >>>
        >>> # Predict using sliding window with 128×128×64 patches
        >>> prediction = sliding_window_inference(
        ...     model=model,
        ...     volume=volume,
        ...     patch_size=(128, 128, 64),
        ...     overlap=0.5,
        ...     batch_size=4,
        ...     device="cuda"
        ... )
        >>> print(prediction.shape)  # (1, 1, 320, 320, 128) - same as input
    """
    from monai.inferers import sliding_window_inference as monai_sliding_window

    # Ensure model is in eval mode
    model.eval()

    # Handle input shape
    squeeze_output = False
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)  # Add batch dimension
        squeeze_output = True

    batch, channels, d, h, w = volume.shape
    if batch != 1:
        raise ValueError(f"Batch size must be 1, got {batch}")

    # Move to device
    volume = volume.to(device)

    # Print info
    # print(f"Sliding window inference on volume shape: {volume.shape}")
    # print(f"Patch size: {patch_size}, Overlap: {overlap:.1%}, Batch size: {batch_size}")

    # Use MONAI's sliding window inference
    with torch.no_grad():
        output = monai_sliding_window(
            inputs=volume,
            roi_size=patch_size,
            sw_batch_size=batch_size,
            predictor=model,
            overlap=overlap,
            mode=blend_mode,  # 'gaussian' or 'constant'
            progress=progress,
        )

    # Remove batch dimension if input didn't have it
    if squeeze_output:
        output = output.squeeze(0)

    return output


def predict_full_volume(
    model: nn.Module,
    input_path: str,
    output_path: str,
    patch_size: tuple = (128, 128, 64),
    overlap: float = 0.5,
    batch_size: int = 4,
    device: str = "cuda",
    target_spacing: Optional[list] = None,
) -> None:
    """
    Predict on a full NIfTI volume using sliding window inference and save result.

    High-level wrapper that loads an input NIfTI file, applies padding if needed,
    runs sliding window inference, and saves the prediction as a NIfTI file with
    the same metadata (affine, header) as the input.

    Args:
        model: Trained PyTorch model (will be set to eval mode)
        input_path: Path to input NIfTI file (.nii or .nii.gz)
        output_path: Path to save output NIfTI file (.nii or .nii.gz)
        patch_size: Size of patches for inference (D, H, W)
        overlap: Overlap ratio between patches [0, 1)
        batch_size: Number of patches to process simultaneously
        device: Device for inference ('cuda' or 'cpu')
        target_spacing: Optional target voxel spacing [x, y, z] in mm
            If provided, input will be resampled before inference

    Example:
        >>> from src.utils import load_unet3d_from_checkpoint, predict_full_volume
        >>> model, _ = load_unet3d_from_checkpoint("model.pth", device="cuda")
        >>>
        >>> predict_full_volume(
        ...     model=model,
        ...     input_path="input_lr.nii.gz",
        ...     output_path="output_sr.nii.gz",
        ...     patch_size=(128, 128, 64),
        ...     overlap=0.5,
        ...     batch_size=4,
        ...     device="cuda"
        ... )
    """
    import nibabel as nib
    from monai.transforms import (
        LoadImage,
        EnsureChannelFirst,
        Orientation,
        Spacing,
        Compose,
    )

    print(f"Loading input volume: {input_path}")

    # Build preprocessing pipeline
    transforms = [
        LoadImage(image_only=False),  # Keep metadata
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

    # Ensure shape is (C, D, H, W)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)

    print(f"Input volume shape: {volume.shape}")

    # Normalize to [0, 1] for consistency with training
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Pad to ensure divisibility by patch size if needed
    original_shape = volume.shape[1:]  # (D, H, W)
    padded_volume, pad_before, orig_shape = pad_to_multiple_of_32(volume.squeeze(0).numpy())
    padded_volume = torch.from_numpy(padded_volume).unsqueeze(0).float()

    print(f"Padded volume shape: {padded_volume.shape}")

    # Run sliding window inference using MONAI
    print(f"Running sliding window inference...")
    prediction = sliding_window_inference(
        model=model,
        volume=padded_volume,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        blend_mode="gaussian",
        progress=True,
    )

    # Remove padding
    prediction = prediction.squeeze(0).cpu().numpy()  # (C, D, H, W) -> (D, H, W)
    if prediction.ndim == 4:
        prediction = prediction.squeeze(0)  # Remove channel dim
    prediction = unpad_volume(prediction, pad_before, orig_shape)

    print(f"Output volume shape: {prediction.shape}")

    # Save as NIfTI with original metadata
    affine = metadata.get("affine", np.eye(4))
    output_nii = nib.Nifti1Image(prediction, affine=affine)

    # Copy header metadata if available
    if "original_affine" in metadata:
        output_nii.header["pixdim"] = metadata.get("pixdim", output_nii.header["pixdim"])

    nib.save(output_nii, output_path)
    print(f"Saved prediction to: {output_path}")
