"""
Diffusion model using Hugging Face's UNet3DConditionModel for 3D medical image super-resolution.

This module uses a hybrid conditioning approach optimized for super-resolution:

## Conditioning Strategy:
1. **LR Image (Concatenation)**: The low-resolution image is concatenated channel-wise
   with the noisy HR image. This provides direct spatial alignment, which is crucial for
   super-resolution tasks.

2. **Resolution (Cross-Attention)**: Resolution parameters are encoded and passed via
   cross-attention. This allows the model to adapt its denoising strategy based on the
   input resolution without affecting spatial alignment.

## Why This Approach?
- **Concatenation for LR**: Ensures pixel-perfect spatial correspondence between LR and HR
- **Cross-Attention for Resolution**: Global parameter that doesn't need spatial alignment
- **Best of Both Worlds**: Combines the efficiency of concatenation with the flexibility
  of cross-attention where each is most effective

This is similar to approaches used in ControlNet and other image-to-image diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import transformers
from diffusers import UNet3DConditionModel, DDPMScheduler, DDIMScheduler


class ResolutionEncoder(nn.Module):
    """
    Encodes resolution parameters into encoder_hidden_states for cross-attention.

    The LR image is now concatenated directly with the noisy HR image,
    so this only handles resolution conditioning via cross-attention.

    Args:
        hidden_dim: Hidden dimension for encoding
        cross_attention_dim: Dimension for cross-attention (must match UNet config)
        num_res_tokens: Number of resolution tokens (default: 8 for richer representation)
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        cross_attention_dim: int = 512,
        num_res_tokens: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_res_tokens = num_res_tokens

        # Resolution embedding network
        self.resolution_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Expand resolution embedding to multiple tokens for richer cross-attention
        self.token_expansion = nn.Linear(hidden_dim, hidden_dim * num_res_tokens)

        # Project to cross-attention dimension
        self.to_cross_attn = nn.Linear(hidden_dim, cross_attention_dim)

    def forward(
        self,
        resolution: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode resolution into cross-attention features.

        Args:
            resolution: Resolution parameters (B, 3)

        Returns:
            Encoded features (B, num_res_tokens, cross_attention_dim)
        """
        batch_size = resolution.shape[0]

        # Encode resolution
        res_emb = self.resolution_embed(resolution)  # (B, hidden_dim)

        # Expand to multiple tokens
        res_tokens = self.token_expansion(res_emb)  # (B, hidden_dim * num_res_tokens)
        res_tokens = res_tokens.reshape(batch_size, self.num_res_tokens, self.hidden_dim)  # (B, num_res_tokens, hidden_dim)

        # Project to cross-attention dimension
        features = self.to_cross_attn(res_tokens)  # (B, num_res_tokens, cross_attention_dim)

        return features


class DiffusionSuperResolution(nn.Module):
    """
    Diffusion-based super-resolution using Hugging Face's UNet3DConditionModel.

    Uses a hybrid conditioning approach:
    - LR image: Concatenated with noisy HR image (channel-wise) for direct spatial alignment
    - Resolution: Cross-attention conditioning for global resolution information

    Args:
        in_channels: Number of input channels for HR image (1 for grayscale MRI)
        out_channels: Number of output channels (1 for predicted noise)
        block_out_channels: Channel counts for each UNet block
        cross_attention_dim: Dimension for cross-attention conditioning
        layers_per_block: Number of layers in each block
        num_frames: Number of frames in 3D volume (depth dimension)
        scheduler_type: Type of scheduler ('ddpm' or 'ddim')
        num_train_timesteps: Number of diffusion timesteps for training
        beta_schedule: Type of beta schedule ('linear', 'scaled_linear', 'squaredcos_cap_v2')
        attention_head_dim: Dimension of attention heads
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        cross_attention_dim: int = 512,
        layers_per_block: int = 2,
        num_frames: int = 16,
        scheduler_type: str = "ddpm",
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim
        self.scheduler_type = scheduler_type

        # Create resolution encoder for cross-attention conditioning
        self.resolution_encoder = ResolutionEncoder(
            hidden_dim=256,
            cross_attention_dim=cross_attention_dim,
            num_res_tokens=8,
        )

        # Create UNet3DConditionModel from diffusers
        # Note: in_channels is now 2 (noisy HR + LR concatenated)
        self.unet = UNet3DConditionModel(
            sample_size=(num_frames, 64, 64),  # Will be overridden by actual input size
            in_channels=in_channels * 2,  # Concatenate noisy HR + LR
            out_channels=out_channels,
            down_block_types=(
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            up_block_types=(
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )

        # Create noise scheduler
        if scheduler_type == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
            )
        elif scheduler_type == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def forward(
        self,
        noisy_sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        resolution: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the UNet3DConditionModel.

        Args:
            noisy_sample: Noisy high-resolution image (B, C, F, H, W)
            timestep: Timesteps (B,) or scalar
            condition: Low-resolution condition image (B, C, F, H, W)
            resolution: Resolution parameters (B, 3) - optional
            return_dict: Whether to return a dict

        Returns:
            Predicted noise (B, C, F, H, W)
        """
        # Concatenate LR condition with noisy HR (channel-wise)
        # This provides direct spatial alignment for super-resolution
        x = torch.cat([noisy_sample, condition], dim=1)  # (B, 2*C, F, H, W)

        # Encode resolution for cross-attention (if provided)
        if resolution is not None:
            encoder_hidden_states = self.resolution_encoder(resolution)
        else:
            # Create dummy encoder states if no resolution provided
            batch_size = noisy_sample.shape[0]
            encoder_hidden_states = torch.zeros(
                batch_size, 8, self.cross_attention_dim,
                device=noisy_sample.device, dtype=noisy_sample.dtype
            )

        # Forward through UNet
        output = self.unet(
            sample=x,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            return output.sample
        else:
            return output[0] if isinstance(output, tuple) else output

    def get_loss(
        self,
        hr_image: torch.Tensor,
        lr_condition: torch.Tensor,
        resolution: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion training loss.

        Args:
            hr_image: High-resolution target image (B, C, F, H, W)
            lr_condition: Low-resolution condition image (B, C, F, H, W)
            resolution: Resolution parameters (B, 3)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (loss, predicted_noise)
        """
        batch_size = hr_image.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=hr_image.device,
        ).long()

        # Add noise to HR image
        if noise is None:
            noise = torch.randn_like(hr_image)

        noisy_hr = self.noise_scheduler.add_noise(hr_image, noise, timesteps)

        # Predict noise
        predicted_noise = self.forward(
            noisy_sample=noisy_hr,
            timestep=timesteps,
            condition=lr_condition,
            resolution=resolution,
        )

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)

        return loss, predicted_noise

    @torch.no_grad()
    def sample(
        self,
        lr_condition: torch.Tensor,
        resolution: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,  # For DDIM
    ) -> torch.Tensor:
        """
        Generate high-resolution image from low-resolution condition.

        Args:
            lr_condition: Low-resolution condition image (B, C, F, H, W)
            resolution: Resolution parameters (B, 3)
            num_inference_steps: Number of denoising steps
            generator: Random generator for reproducibility
            eta: DDIM eta parameter (0.0 = deterministic, 1.0 = DDPM)

        Returns:
            Generated high-resolution image (B, C, F, H, W)
        """
        batch_size = lr_condition.shape[0]
        device = lr_condition.device
        shape = lr_condition.shape

        # Start from random noise
        image = torch.randn(shape, device=device, generator=generator)

        # Set number of inference steps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        # Encode resolution once (if provided, it doesn't change during sampling)
        if resolution is not None:
            encoder_hidden_states = self.resolution_encoder(resolution)
        else:
            encoder_hidden_states = torch.zeros(
                batch_size, 8, self.cross_attention_dim,
                device=device, dtype=lr_condition.dtype
            )

        # Denoise iteratively
        for t in self.noise_scheduler.timesteps:
            # Concatenate current noisy image with LR condition
            x = torch.cat([image, lr_condition], dim=1)  # (B, 2*C, F, H, W)

            # Predict noise
            noise_pred = self.unet(
                sample=x,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            # Compute previous sample
            if self.scheduler_type == "ddim":
                # DDIM step with eta parameter
                image = self.noise_scheduler.step(
                    noise_pred, t, image, eta=eta
                ).prev_sample
            else:
                # DDPM step
                image = self.noise_scheduler.step(noise_pred, t, image).prev_sample

        return image

    def enable_xformers_memory_efficient_attention(self):
        """Enable memory efficient attention from xformers."""
        self.unet.enable_xformers_memory_efficient_attention()

    def disable_xformers_memory_efficient_attention(self):
        """Disable memory efficient attention from xformers."""
        self.unet.disable_xformers_memory_efficient_attention()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for reduced memory usage during training."""
        self.unet.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.unet.disable_gradient_checkpointing()


def create_diffusion_model(
    image_size: Tuple[int, int, int] = (128, 128, 128),
    in_channels: int = 1,
    model_size: str = "base",
    scheduler_type: str = "ddpm",
    num_train_timesteps: int = 1000,
    beta_schedule: str = "linear",
) -> DiffusionSuperResolution:
    """
    Factory function to create diffusion model with preset configurations.

    Args:
        image_size: Input image size (D, H, W)
        in_channels: Number of input channels
        model_size: Model size preset ('tiny', 'small', 'base', 'large')
        scheduler_type: Type of scheduler ('ddpm' or 'ddim')
        num_train_timesteps: Number of diffusion timesteps
        beta_schedule: Beta schedule type

    Returns:
        Configured DiffusionSuperResolution model
    """
    # Model size presets
    configs = {
        "tiny": {
            "block_out_channels": (64, 128, 256, 256),
            "cross_attention_dim": 256,
            "attention_head_dim": 4,
        },
        "small": {
            "block_out_channels": (96, 192, 384, 384),
            "cross_attention_dim": 384,
            "attention_head_dim": 6,
        },
        "base": {
            "block_out_channels": (128, 256, 512, 512),
            "cross_attention_dim": 512,
            "attention_head_dim": 8,
        },
        "large": {
            "block_out_channels": (160, 320, 640, 640),
            "cross_attention_dim": 640,
            "attention_head_dim": 10,
        },
    }

    if model_size not in configs:
        raise ValueError(
            f"Unknown model_size: {model_size}. Choose from {list(configs.keys())}"
        )

    config = configs[model_size]

    return DiffusionSuperResolution(
        in_channels=in_channels,
        out_channels=in_channels,
        block_out_channels=config["block_out_channels"],
        cross_attention_dim=config["cross_attention_dim"],
        layers_per_block=2,
        num_frames=image_size[0],  # Depth dimension
        scheduler_type=scheduler_type,
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        attention_head_dim=config["attention_head_dim"],
    )


if __name__ == "__main__":
    # Test the new hybrid conditioning approach
    print("=" * 80)
    print("Testing Diffusion Model with Hybrid Conditioning")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    print("\n1. Creating model (base size)...")
    model = create_diffusion_model(
        image_size=(64, 64, 64),  # Smaller for testing
        in_channels=1,
        model_size="tiny",  # Use tiny for faster testing
        scheduler_type="ddpm",
    ).to(device)

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\n2. Testing forward pass (training)...")
    batch_size = 2
    hr_image = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    lr_image = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    resolution = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).to(device)

    # Compute loss
    loss, predicted_noise = model.get_loss(hr_image, lr_image, resolution)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Predicted noise shape: {predicted_noise.shape}")

    # Test sampling
    print("\n3. Testing sampling (inference)...")
    with torch.no_grad():
        generated = model.sample(
            lr_condition=lr_image[:1],  # Generate for one sample
            resolution=resolution[:1],
            num_inference_steps=10,  # Few steps for testing
        )
    print(f"   Generated image shape: {generated.shape}")
    print(f"   Generated image range: [{generated.min():.3f}, {generated.max():.3f}]")

    # Test without resolution conditioning
    print("\n4. Testing without resolution conditioning...")
    with torch.no_grad():
        generated_no_res = model.sample(
            lr_condition=lr_image[:1],
            resolution=None,  # No resolution
            num_inference_steps=10,
        )
    print(f"   Generated image shape: {generated_no_res.shape}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("\nKey improvements with concatenation conditioning:")
    print("  • Direct spatial alignment between LR and HR")
    print("  • More efficient than pure cross-attention")
    print("  • Resolution still conditioned via cross-attention")
    print("  • Better suited for super-resolution tasks")
    print("=" * 80)
