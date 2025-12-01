"""
Multiple MONAI model architectures for 3D medical image super-resolution regression.

This module provides a registry of different MONAI architectures that can be easily
switched during training to find the best performing model.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from monai.networks.nets import (
    SwinUNETR,
    UNETR,
    SegResNet,
    UNet,
    AttentionUnet,
    VNet,
    DynUNet,
    BasicUNet,
)


def get_swinunetr(
    img_size: Tuple[int, int, int] = (128, 128, 128),
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 48,
    use_checkpoint: bool = False,
    spatial_dims: int = 3,
) -> nn.Module:
    """
    Swin UNETR - Transformer-based U-Net with Swin Transformer encoder.

    Good for: Capturing long-range dependencies, high performance
    Cons: High memory usage, slower training

    Args:
        img_size: Input image size (D, H, W)
        in_channels: Number of input channels
        out_channels: Number of output channels
        feature_size: Base feature size (24, 48, 96)
        use_checkpoint: Use gradient checkpointing to save memory
        spatial_dims: Number of spatial dimensions (3 for 3D)
    """
    return SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
    )


def get_unetr(
    img_size: Tuple[int, int, int] = (128, 128, 128),
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    pos_embed: str = "conv",
    norm_name: str = "instance",
    dropout_rate: float = 0.0,
    spatial_dims: int = 3,
) -> nn.Module:
    """
    UNETR - Vision Transformer-based U-Net.

    Good for: Capturing global context, attention mechanisms
    Cons: Memory intensive, requires larger datasets

    Args:
        img_size: Input image size (D, H, W)
        in_channels: Number of input channels
        out_channels: Number of output channels
        feature_size: Feature size for decoder (8, 16, 32)
        hidden_size: Hidden size for transformer (768, 1024)
        mlp_dim: MLP dimension in transformer
        num_heads: Number of attention heads (12, 16)
        pos_embed: Position embedding type ('conv' or 'perceptron')
        norm_name: Normalization type ('instance', 'batch', 'group')
        dropout_rate: Dropout rate
        spatial_dims: Number of spatial dimensions (3 for 3D)
    """
    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        pos_embed=pos_embed,
        norm_name=norm_name,
        dropout_rate=dropout_rate,
        spatial_dims=spatial_dims,
    )


def get_segresnet(
    in_channels: int = 1,
    out_channels: int = 1,
    init_filters: int = 32,
    blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
    blocks_up: Tuple[int, ...] = (1, 1, 1),
    dropout_prob: Optional[float] = None,
    spatial_dims: int = 3,
) -> nn.Module:
    """
    SegResNet - Residual U-Net with efficient design.

    Good for: Fast training, low memory, good performance
    Recommended: START HERE - often best balance of speed/performance

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_filters: Initial number of filters (16, 32, 64)
        blocks_down: Number of residual blocks in each encoder level
        blocks_up: Number of residual blocks in each decoder level
        dropout_prob: Dropout probability
        spatial_dims: Number of spatial dimensions (3 for 3D)
    """
    return SegResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
        spatial_dims=spatial_dims,
    )


def get_unet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: str = "batch",
    dropout: float = 0.0,
) -> nn.Module:
    """
    Standard MONAI U-Net with configurable depth.

    Good for: Baseline, reliable performance, flexible

    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: Sequence of channels for each level
        strides: Strides for each downsampling level
        num_res_units: Number of residual units per level (0-4)
        norm: Normalization type ('batch', 'instance', 'group')
        dropout: Dropout probability
    """
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )


def get_attention_unet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    dropout: float = 0.0,
) -> nn.Module:
    """
    Attention U-Net with attention gates.

    Good for: Feature refinement, boundary detection
    Cons: Slower than standard U-Net

    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        channels: Sequence of channels for each level
        strides: Strides for each downsampling level
        dropout: Dropout probability
    """
    return AttentionUnet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        dropout=dropout,
    )


def get_vnet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    dropout_prob: float = 0.0,
    dropout_dim: int = 3,
    act: str = "elu",
    bias: bool = False,
) -> nn.Module:
    """
    V-Net architecture with residual connections.

    Good for: Medical imaging, residual learning

    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout_prob: Dropout probability
        dropout_dim: Dropout dimension (1, 2, or 3)
        act: Activation function ('elu', 'relu', 'prelu')
        bias: Use bias in convolutions
    """
    return VNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout_prob,
        dropout_dim=dropout_dim,
        act=act,
        bias=bias,
    )


def get_basicunet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    features: Tuple[int, ...] = (32, 32, 64, 128, 256, 32),
    act: str = "relu",
    norm: str = "batch",
    bias: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Basic U-Net - Simple and lightweight.

    Good for: Quick experiments, limited memory
    Cons: Less powerful than other architectures

    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        features: Feature sizes for encoder, bottleneck, and decoder
        act: Activation function ('relu', 'prelu', 'elu')
        norm: Normalization type ('batch', 'instance', 'group')
        bias: Use bias in convolutions
        dropout: Dropout probability
    """
    return BasicUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        act=act,
        norm=norm,
        bias=bias,
        dropout=dropout,
    )


def get_dynunet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    kernel_size: Tuple[Tuple[int, ...], ...] = ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    strides: Tuple[Tuple[int, ...], ...] = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    upsample_kernel_size: Tuple[Tuple[int, ...], ...] = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    filters: Tuple[int, ...] = (32, 64, 128, 256, 512),
    dropout: Optional[float] = 0.0,
    norm_name: str = "instance",
    act_name: str = "leakyrelu",
    deep_supervision: bool = False,
    deep_supr_num: int = 1,
    res_block: bool = True,
) -> nn.Module:
    """
    Dynamic U-Net - Highly configurable U-Net.

    Good for: Custom architectures, flexibility

    Args:
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel sizes for each level
        strides: Strides for each level
        upsample_kernel_size: Upsampling kernel sizes
        filters: Number of filters for each level
        dropout: Dropout probability
        norm_name: Normalization type ('instance', 'batch', 'group')
        act_name: Activation function ('leakyrelu', 'relu', 'prelu')
        deep_supervision: Use deep supervision
        deep_supr_num: Number of deep supervision levels
        res_block: Use residual blocks
    """
    return DynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        dropout=dropout,
        norm_name=norm_name,
        act_name=act_name,
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
        res_block=res_block,
    )


# Model registry for easy access
MODEL_REGISTRY = {
    "swinunetr": get_swinunetr,
    "unetr": get_unetr,
    "segresnet": get_segresnet,
    "unet": get_unet,
    "attention_unet": get_attention_unet,
    "vnet": get_vnet,
    "basicunet": get_basicunet,
    "dynunet": get_dynunet,
}


# Preset configurations for different model sizes
MODEL_PRESETS = {
    # SegResNet presets (recommended starting point)
    "segresnet_small": {
        "model_type": "segresnet",
        "init_filters": 16,
        "blocks_down": (1, 2, 2, 4),
        "blocks_up": (1, 1, 1),
    },
    "segresnet_base": {
        "model_type": "segresnet",
        "init_filters": 32,
        "blocks_down": (1, 2, 2, 4),
        "blocks_up": (1, 1, 1),
    },
    "segresnet_large": {
        "model_type": "segresnet",
        "init_filters": 64,
        "blocks_down": (1, 2, 2, 4, 4),
        "blocks_up": (1, 1, 1, 1),
    },

    # SwinUNETR presets
    "swinunetr_small": {
        "model_type": "swinunetr",
        "feature_size": 24,
        "use_checkpoint": False,
    },
    "swinunetr_base": {
        "model_type": "swinunetr",
        "feature_size": 48,
        "use_checkpoint": False,
    },
    "swinunetr_large": {
        "model_type": "swinunetr",
        "feature_size": 96,
        "use_checkpoint": True,  # Use checkpointing for memory
    },

    # UNETR presets
    "unetr_small": {
        "model_type": "unetr",
        "feature_size": 8,
        "hidden_size": 512,
        "mlp_dim": 2048,
        "num_heads": 8,
    },
    "unetr_base": {
        "model_type": "unetr",
        "feature_size": 16,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
    },

    # UNet presets
    "unet_small": {
        "model_type": "unet",
        "channels": (16, 32, 64, 128),
        "strides": (2, 2, 2),
        "num_res_units": 2,
    },
    "unet_base": {
        "model_type": "unet",
        "channels": (32, 64, 128, 256, 512),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    },
}


def create_model(
    model_name: str,
    img_size: Tuple[int, int, int] = (128, 128, 128),
    in_channels: int = 1,
    out_channels: int = 1,
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_name: Name of the model or preset (e.g., 'segresnet', 'segresnet_base')
        img_size: Input image size (D, H, W)
        in_channels: Number of input channels
        out_channels: Number of output channels
        device: Device to move model to (cuda/cpu)
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model

    Examples:
        >>> # Create a SegResNet model (recommended)
        >>> model = create_model('segresnet', img_size=(128, 128, 128))

        >>> # Create a preset model
        >>> model = create_model('segresnet_large', img_size=(128, 128, 128))

        >>> # Create a SwinUNETR with custom settings
        >>> model = create_model('swinunetr', img_size=(96, 96, 96), feature_size=48)
    """
    # Check if it's a preset
    if model_name in MODEL_PRESETS:
        preset = MODEL_PRESETS[model_name].copy()
        actual_model_type = preset.pop("model_type")
        # Merge preset config with kwargs (kwargs take precedence)
        preset.update(kwargs)
        kwargs = preset
        model_name = actual_model_type

    # Get model constructor
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys()) + list(MODEL_PRESETS.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )

    model_fn = MODEL_REGISTRY[model_name]

    # Add common arguments
    common_args = {
        "in_channels": in_channels,
        "out_channels": out_channels,
    }

    # Add img_size for models that need it
    if model_name in ["swinunetr", "unetr"]:
        common_args["img_size"] = img_size

    # Merge with user kwargs
    common_args.update(kwargs)

    # Create model
    model = model_fn(**common_args)

    # Move to device if specified
    if device is not None:
        model = model.to(device)

    return model


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model_name: Name of the model or preset

    Returns:
        Dictionary with model information
    """
    info = {
        "name": model_name,
        "available": model_name in MODEL_REGISTRY or model_name in MODEL_PRESETS,
    }

    if model_name in MODEL_PRESETS:
        info["type"] = "preset"
        info["config"] = MODEL_PRESETS[model_name]
    elif model_name in MODEL_REGISTRY:
        info["type"] = "base_model"

    return info


def list_available_models():
    """Print all available models and presets."""
    print("=" * 80)
    print("Available Base Models:")
    print("=" * 80)
    for name in MODEL_REGISTRY.keys():
        print(f"  - {name}")

    print("\n" + "=" * 80)
    print("Available Presets:")
    print("=" * 80)
    for name, config in MODEL_PRESETS.items():
        print(f"  - {name}")
        print(f"    Config: {config}")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing MONAI Model Registry")
    print("=" * 80)

    # List all available models
    list_available_models()

    # Test creating different models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nCreating test models...")

    # Test SegResNet (recommended)
    print("\n1. SegResNet (recommended starting point):")
    model = create_model("segresnet_base", img_size=(128, 128, 128), device=device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with random input
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"   Input shape: {x.shape}, Output shape: {y.shape}")

    # Test SwinUNETR
    print("\n2. SwinUNETR:")
    model = create_model("swinunetr_small", img_size=(96, 96, 96), device=device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test UNet
    print("\n3. Standard UNet:")
    model = create_model("unet_base", img_size=(128, 128, 128), device=device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 80)
    print("All tests passed!")
