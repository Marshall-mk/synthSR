import os
import argparse
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from typing import List
import wandb

# MONAI imports
from monai.data import DataLoader

# Import our modules
from src.model import UNet3D
from src.utils import (
    save_model_checkpoint,
    get_image_paths,
    save_training_config,
    find_latest_checkpoint,
    get_loss_function,
    calculate_metrics,
    sliding_window_inference,
)
from src.data_fft import HRLRDataGenerator, create_dataset


def train_regression_model(
    hr_image_paths: List[str],
    model_dir: str,
    epochs: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    loss_name: str = "l1",
    output_shape: tuple = (128, 128, 128),
    checkpoint: str = None,
    device: str = "cuda",
    save_interval: int = 10,
    val_image_paths: List[str] = None,
    atlas_res: list = [1.0, 1.0, 1.0],
    min_resolution: list = [1.0, 1.0, 1.0],
    max_res_aniso: list = [9.0, 9.0, 9.0],
    randomise_res: bool = True,
    apply_lr_deformation: bool = True,
    apply_bias_field: bool = True,
    apply_intensity_aug: bool = True,
    enable_90_rotations: bool = False,
    clip_to_unit_range: bool = True,
    num_workers: int = None,
    use_cache: bool = False,
    use_sliding_window_val: bool = False,
    sw_overlap: float = 0.5,
    sw_batch_size: int = 4,
    use_wandb: bool = False,
    wandb_project: str = "synthsr",
    wandb_entity: str = None,
    wandb_run_name: str = None,
):
    """
    Stage 1: Train regression UNet

    Args:
        hr_image_paths: List of paths to high-resolution images
        model_dir: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_shape: Output volume shape
        checkpoint: Optional checkpoint to resume from
        device: 'cuda' or 'cpu'
        save_interval: Save checkpoint every N epochs
        val_image_paths: Optional list of validation image paths
        atlas_res: Physical resolution of input HR images [x, y, z] in mm
        min_resolution: Minimum resolution for randomization
        max_res_aniso: Maximum anisotropic resolution
        randomise_res: Whether to randomize resolution
        apply_lr_deformation: Whether to apply deformation to LR
        apply_bias_field: Whether to apply bias field
        apply_intensity_aug: Whether to apply intensity augmentation
        enable_90_rotations: Enable 90° rotations
        clip_to_unit_range: Whether to clip images to [0, 1] range
        num_workers: Number of data loading workers
        use_cache: Whether to use CacheDataset
        use_sliding_window_val: If True, use sliding window inference for validation
            to get accurate full-volume metrics. Slower but more representative.
        sw_overlap: Overlap for sliding window validation (default: 0.5)
        sw_batch_size: Batch size for sliding window patches (default: 4)
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: W&B project name (default: "synthsr")
        wandb_entity: W&B entity/team name (optional)
        wandb_run_name: W&B run name (optional, auto-generated if None)
    """
    print("=" * 80)
    print("STAGE 1: Training Regression UNet")
    print("=" * 80)

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Auto-detect optimal num_workers if not provided
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        if device == "cuda" and torch.cuda.is_available():
            num_workers = min(4, max(cpu_count // 2, 1))
        elif cpu_count >= 4:
            num_workers = 2
        else:
            num_workers = 0

    # Enable pin_memory for faster GPU transfer
    pin_memory = device == "cuda" and torch.cuda.is_available() and num_workers > 0

    print(
        f"DataLoader settings: num_workers={num_workers}, pin_memory={pin_memory}, use_cache={use_cache}"
    )

    # Create data generator
    print(f"Input HR image resolution: {atlas_res} mm")
    print(f"Resolution randomization: {min_resolution} to {max_res_aniso} mm")
    print(
        f"LR deformation: {apply_lr_deformation}, "
        f"Bias field: {apply_bias_field}, Intensity aug: {apply_intensity_aug}"
    )

    generator = HRLRDataGenerator(
        atlas_res=atlas_res,
        target_res=[1.0, 1.0, 1.0],
        output_shape=list(output_shape),
        min_resolution=min_resolution,
        max_res_aniso=max_res_aniso,
        randomise_res=randomise_res,
        apply_lr_deformation=apply_lr_deformation,
        apply_bias_field=apply_bias_field,
        apply_intensity_aug=apply_intensity_aug,
        enable_90_rotations=enable_90_rotations,
        clip_to_unit_range=clip_to_unit_range,  # Ensure [0, 1] range
    )

    # Create dataset
    dataset = create_dataset(
        image_paths=hr_image_paths,
        generator=generator,
        target_shape=list(output_shape),
        target_spacing=atlas_res,
        use_cache=use_cache,
        return_resolution=True,  # Return resolution for logging
        is_training=True,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    print(f"Training dataset: {len(dataset)} images")

    # Create validation dataset if provided
    val_dataloader = None
    if val_image_paths:
        val_dataset = create_dataset(
            image_paths=val_image_paths,
            generator=generator,
            target_shape=list(output_shape),
            target_spacing=atlas_res,
            use_cache=use_cache,
            return_resolution=True,  # Return resolution for logging
            is_training=False,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        print(f"Validation dataset: {len(val_dataset)} images")

    # Create regression UNet
    model = UNet3D(
        nb_features=24,
        input_shape=(1, *output_shape),
        nb_levels=5,
        conv_size=3,
        nb_labels=1,
        feat_mult=2,
        final_pred_activation="linear",
        nb_conv_per_level=2,
    )

    # Move model to device
    model = model.to(device)

    # Auto-detect and load checkpoint
    start_epoch = 0
    checkpoint_data = None

    # If no checkpoint specified, try to find the latest one automatically
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(model_dir)
        if checkpoint:
            print(f"Auto-detected checkpoint: {checkpoint}")

    # Load checkpoint if available
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        start_epoch = checkpoint_data.get("epoch", 0) + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = get_loss_function(loss_name)
    print(f"Using loss function: {loss_name}")

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7
    )

    # Load optimizer and scheduler state if resuming
    if checkpoint_data is not None:
        if "optimizer_state_dict" in checkpoint_data:
            print("Loading optimizer state...")
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint_data:
            print("Loading scheduler state...")
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {epochs} epochs, batch size {batch_size}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Device: {device}")

    # Initialize Weights & Biases if enabled
    if use_wandb:
        wandb_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_function": loss_name,
            "output_shape": output_shape,
            "atlas_res": atlas_res,
            "min_resolution": min_resolution,
            "max_res_aniso": max_res_aniso,
            "randomise_res": randomise_res,
            "apply_lr_deformation": apply_lr_deformation,
            "apply_bias_field": apply_bias_field,
            "apply_intensity_aug": apply_intensity_aug,
            "enable_90_rotations": enable_90_rotations,
            "clip_to_unit_range": clip_to_unit_range,
            "num_workers": num_workers,
            "use_cache": use_cache,
            "use_sliding_window_val": use_sliding_window_val,
            "sw_overlap": sw_overlap,
            "sw_batch_size": sw_batch_size,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "n_train_samples": len(hr_image_paths),
            "n_val_samples": len(val_image_paths) if val_image_paths else 0,
        }

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            resume="allow" if checkpoint else False,
        )

        # Watch model gradients and parameters
        wandb.watch(model, criterion, log="all", log_freq=100)
        print(f"Weights & Biases initialized: {wandb.run.name}")

    # Setup CSV logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_log_{timestamp}.csv"
    csv_path = os.path.join(model_dir, csv_filename)
    csv_exists = os.path.exists(csv_path)

    # Define CSV headers
    csv_headers = ["epoch", "loss_type", "train_loss", "learning_rate", "epoch_time", "data_loading_time", "forward_backward_time"]
    if val_dataloader:
        csv_headers.extend(["val_loss", "val_mae", "val_mse", "val_rmse", "val_psnr", "val_r2", "validation_time"])

    # Open CSV file for writing (append if resuming training)
    csv_file = open(csv_path, mode='a', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)

    # Write header if file is new
    if not csv_exists or os.path.getsize(csv_path) == 0:
        csv_writer.writeheader()
        csv_file.flush()

    print(f"Logging training metrics to: {csv_path}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0

        # Timing accumulators
        data_loading_time = 0.0
        forward_backward_time = 0.0
        batch_start_time = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (input_img, target_img, resolution, thickness) in enumerate(pbar):
            # Data loading time
            data_loading_time += time.time() - batch_start_time

            model_start_time = time.time()

            # Move to device
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            resolution = resolution.to(device)
            thickness = thickness.to(device)

            # # Log shapes and resolution every 50 batches
            # # if batch_idx % 50 == 0:
            # print(f"\nBatch {batch_idx}: Input shape: {input_img.shape}, Target shape: {target_img.shape}")
            # print(f"Resolution: {resolution.cpu().numpy().flatten()}, Thickness: {thickness.cpu().numpy().flatten()}")
            # print(f"Input range: [{input_img.min().item():.4f}, {input_img.max().item():.4f}], Target range: [{target_img.min().item():.4f}, {target_img.max().item():.4f}]")

            # # Save intermediate input and target images for debugging
            # import nibabel as nib
            # import numpy as np

            # # Save each image in the batch separately
            # for b in range(input_img.shape[0]):
            #     # Index batch first, then remove channel dimension
            #     input_img_np = input_img[b].squeeze(0).cpu().numpy()  # Shape: (128, 128, 128)
            #     target_img_np = target_img[b].squeeze(0).cpu().numpy()  # Shape: (128, 128, 128)

            #     input_nii = nib.Nifti1Image(input_img_np, np.eye(4))
            #     target_nii = nib.Nifti1Image(target_img_np, np.eye(4))

            #     # Include batch index in filename
            #     nib.save(input_nii, f"debug_input_epoch{epoch+1}_batch{batch_idx+1}_sample{b+1}.nii.gz")
            #     nib.save(target_nii, f"debug_target_epoch{epoch+1}_batch{batch_idx+1}_sample{b+1}.nii.gz")

            # break  # --- DEBUGGING ONLY ---

            # Forward pass
            output = model(input_img)
            loss = criterion(output, target_img)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Forward/backward time
            forward_backward_time += time.time() - model_start_time

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Start timing for next batch data loading
            batch_start_time = time.time()

        avg_loss = epoch_loss / len(dataloader)

        # Validation
        val_loss = None
        val_metrics = None
        validation_time = 0.0
        if val_dataloader:
            val_start_time = time.time()
            model.eval()
            val_epoch_loss = 0.0
            # Accumulate metrics across validation set
            metrics_sum = {
                "mae": 0.0,
                "mse": 0.0,
                "rmse": 0.0,
                "psnr": 0.0,
                "r2": 0.0,
            }
            num_val_batches = 0

            with torch.no_grad():
                if use_sliding_window_val:
                    # Use sliding window inference for accurate full-volume validation
                    print(f"\n  Running sliding window validation...")
                    for input_img, target_img, resolution, thickness in val_dataloader:
                        input_img = input_img.to(device)
                        target_img = target_img.to(device)

                        # Process each sample in batch with sliding window
                        for b in range(input_img.shape[0]):
                            input_sample = input_img[b:b+1]
                            target_sample = target_img[b:b+1]

                            # Run sliding window inference
                            output_sample = sliding_window_inference(
                                model=model,
                                volume=input_sample,
                                patch_size=output_shape,
                                overlap=sw_overlap,
                                batch_size=sw_batch_size,
                                device=device,
                                blend_mode="gaussian",
                                progress=False,
                            )

                            # Calculate loss and metrics
                            loss = criterion(output_sample, target_sample)
                            val_epoch_loss += loss.item()

                            batch_metrics = calculate_metrics(output_sample, target_sample, max_val=1.0)
                            for key in metrics_sum:
                                metrics_sum[key] += batch_metrics[key]
                            num_val_batches += 1
                else:
                    # Standard patch-based validation (faster)
                    for input_img, target_img, resolution, thickness in val_dataloader:
                        input_img = input_img.to(device)
                        target_img = target_img.to(device)
                        output = model(input_img)
                        loss = criterion(output, target_img)
                        val_epoch_loss += loss.item()

                        # Calculate metrics for this batch
                        batch_metrics = calculate_metrics(output, target_img, max_val=1.0)
                        for key in metrics_sum:
                            metrics_sum[key] += batch_metrics[key]
                        num_val_batches += 1

            val_loss = val_epoch_loss / num_val_batches

            # Average metrics across batches
            val_metrics = {k: v / num_val_batches for k, v in metrics_sum.items()}

            validation_time = time.time() - val_start_time

            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}"
            )
            print(
                f"  Val Metrics - MAE: {val_metrics['mae']:.4f} | MSE: {val_metrics['mse']:.6f} | "
                f"RMSE: {val_metrics['rmse']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB | "
                f"R²: {val_metrics['r2']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            print(
                f"  Timing - Epoch: {epoch_time:.1f}s | Data: {data_loading_time:.1f}s | "
                f"Model: {forward_backward_time:.1f}s | Val: {validation_time:.1f}s"
            )
        else:
            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            print(
                f"  Timing - Epoch: {epoch_time:.1f}s | Data: {data_loading_time:.1f}s | "
                f"Model: {forward_backward_time:.1f}s"
            )

        # Update learning rate scheduler
        scheduler.step(val_loss if val_loss is not None else avg_loss)

        # Log metrics to CSV
        log_data = {
            "epoch": epoch + 1,
            "loss_type": loss_name,
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time,
            "data_loading_time": data_loading_time,
            "forward_backward_time": forward_backward_time,
        }

        if val_dataloader and val_loss is not None and val_metrics is not None:
            log_data.update({
                "val_loss": val_loss,
                "val_mae": val_metrics['mae'],
                "val_mse": val_metrics['mse'],
                "val_rmse": val_metrics['rmse'],
                "val_psnr": val_metrics['psnr'],
                "val_r2": val_metrics['r2'],
                "validation_time": validation_time,
            })

        csv_writer.writerow(log_data)
        csv_file.flush()  # Ensure data is written immediately

        # Log to Weights & Biases
        if use_wandb:
            wandb_log_data = {
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "timing/epoch_time": epoch_time,
                "timing/data_loading_time": data_loading_time,
                "timing/forward_backward_time": forward_backward_time,
            }

            if val_dataloader and val_loss is not None and val_metrics is not None:
                wandb_log_data.update({
                    "val/loss": val_loss,
                    "val/mae": val_metrics['mae'],
                    "val/mse": val_metrics['mse'],
                    "val/rmse": val_metrics['rmse'],
                    "val/psnr": val_metrics['psnr'],
                    "val/r2": val_metrics['r2'],
                    "timing/validation_time": validation_time,
                })

            wandb.log(wandb_log_data)

        # Save checkpoint at specified intervals
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                model_dir, f"regression_unet_epoch_{epoch + 1:04d}.pth"
            )
            model_config = {
                "nb_features": 24,
                "input_shape": (1, *output_shape),
                "nb_levels": 5,
                "conv_size": 3,
                "nb_labels": 1,
                "feat_mult": 2,
                "final_pred_activation": "linear",
                "nb_conv_per_level": 2,
            }
            save_model_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                val_loss=val_loss,
                model_type="unet3d",
                model_config=model_config,
                scheduler_state_dict=scheduler.state_dict(),
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    # Close CSV file
    csv_file.close()
    print(f"Training log saved to: {csv_path}")

    # Save final model
    final_path = os.path.join(model_dir, "regression_unet_final.pth")
    model_config = {
        "nb_features": 24,
        "input_shape": (1, *output_shape),
        "nb_levels": 5,
        "conv_size": 3,
        "nb_labels": 1,
        "feat_mult": 2,
        "final_pred_activation": "linear",
        "nb_conv_per_level": 2,
    }
    save_model_checkpoint(
        filepath=final_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs - 1,
        model_type="unet3d",
        model_config=model_config,
        scheduler_state_dict=scheduler.state_dict(),
    )
    print(f"Training complete! Final model saved to: {final_path}")

    # Finish Weights & Biases run
    if use_wandb:
        # Save final model as artifact
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="Final trained SynthSR UNet3D model",
        )
        artifact.add_file(final_path)
        wandb.log_artifact(artifact)
        wandb.finish()
        print("Weights & Biases run finished and model artifact saved")


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train SynthSR++ model with MONAI")

    # Data source arguments
    parser.add_argument(
        "--hr_image_dir",
        type=str,
        default=None,
        help="Directory containing high-resolution images",
    )
    parser.add_argument(
        "--csv_file", type=str, default=None, help="CSV file with image metadata"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory for relative paths in CSV",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory to save trained models"
    )

    # Model and data parameters
    parser.add_argument(
        "--output_shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Output volume shape",
    )
    parser.add_argument(
        "--atlas_res",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Physical resolution of input HR images [x, y, z] in mm",
    )
    parser.add_argument(
        "--min_resolution",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Minimum resolution for randomization [x, y, z] in mm",
    )
    parser.add_argument(
        "--max_res_aniso",
        type=float,
        nargs=3,
        default=[9.0, 9.0, 9.0],
        help="Maximum anisotropic resolution [x, y, z] in mm",
    )

    # Augmentation flags
    parser.add_argument(
        "--no_randomise_res",
        action="store_true",
        help="Disable resolution randomization",
    )
    parser.add_argument(
        "--no_lr_deformation", action="store_true", help="Disable LR deformation"
    )
    parser.add_argument(
        "--no_bias_field", action="store_true", help="Disable bias field corruption"
    )
    parser.add_argument(
        "--no_intensity_aug", action="store_true", help="Disable intensity augmentation"
    )
    parser.add_argument(
        "--enable_90_rotations", action="store_true", help="Enable 90-degree rotations"
    )
    parser.add_argument(
        "--disable_clip", action="store_true", help="Disable clipping to [0, 1] range"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="l1",
        choices=["l1", "l2", "huber", "ssim", "gaussian_nll"],
        help="Loss function: l1 (MAE), l2 (MSE), huber, ssim, or gaussian_nll",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of data loading workers (default: auto-detect)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use MONAI CacheDataset for faster loading (caches preprocessed images)",
    )

    # Validation arguments
    parser.add_argument(
        "--use_sliding_window_val",
        action="store_true",
        help="Use sliding window inference for validation (accurate full-volume metrics, slower)",
    )
    parser.add_argument(
        "--sw_overlap",
        type=float,
        default=0.5,
        help="Overlap ratio for sliding window validation (0.0-0.9, default: 0.5)",
    )
    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=4,
        help="Batch size for sliding window patches during validation (default: 4)",
    )
    parser.add_argument(
        "--val_image_dir",
        type=str,
        default=None,
        help="Optional validation images directory",
    )

    # Weights & Biases arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases tracking",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="synthsr",
        help="W&B project name (default: synthsr)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (optional, auto-generated if not provided)",
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Get training image paths
    hr_image_paths = get_image_paths(
        image_dir=args.hr_image_dir,
        csv_file=args.csv_file,
        base_dir=args.base_dir,
        split="train",
        model_dir=args.model_dir,
    )

    # Get validation image paths if needed
    val_image_paths = None
    if args.val_image_dir or args.csv_file:
        val_image_paths = get_image_paths(
            image_dir=args.val_image_dir,
            csv_file=args.csv_file,
            base_dir=args.base_dir,
            split="val",
            model_dir=args.model_dir,
        )

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Save training configuration
    save_training_config(
        model_dir=args.model_dir,
        args=args,
        n_train_samples=len(hr_image_paths),
        n_val_samples=len(val_image_paths) if val_image_paths else 0,
        training_stage="stage1",
    )

    # Train regression UNet
    train_regression_model(
        hr_image_paths=hr_image_paths,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_name=args.loss,
        output_shape=tuple(args.output_shape),
        checkpoint=args.checkpoint,
        device=args.device,
        save_interval=args.save_interval,
        val_image_paths=val_image_paths,
        atlas_res=args.atlas_res,
        min_resolution=args.min_resolution,
        max_res_aniso=args.max_res_aniso,
        randomise_res=not args.no_randomise_res,
        apply_lr_deformation=not args.no_lr_deformation,
        apply_bias_field=not args.no_bias_field,
        apply_intensity_aug=not args.no_intensity_aug,
        enable_90_rotations=args.enable_90_rotations,
        clip_to_unit_range=not args.disable_clip,
        num_workers=args.num_workers,
        use_cache=args.use_cache,
        use_sliding_window_val=args.use_sliding_window_val,
        sw_overlap=args.sw_overlap,
        sw_batch_size=args.sw_batch_size,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )
