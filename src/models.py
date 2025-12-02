"""
Multiple model architectures for 3D medical image super-resolution regression.

This module provides a unified registry of different architectures including:
- Custom UNet3D (from original SynthSR)
- MONAI architectures (SwinUNETR, UNETR, SegResNet, etc.)

All models can be easily switched during training to find the best performing model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
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


# ============================================================================
# Custom UNet3D Implementation (Original SynthSR Architecture)
# ============================================================================

class ConvBlock(nn.Module):
    """
    Basic convolutional block with optional residual connection, dropout, and batch norm.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        conv_size: Kernel size for convolution
        nb_conv: Number of convolutions in this block
        ndims: Number of spatial dimensions (2 or 3)
        padding: Padding mode ('same' or 'valid')
        dilation_rate: Dilation rate for convolution
        activation: Activation function name ('elu', 'relu', 'leakyrelu', etc.)
        use_residuals: Whether to use residual connections
        conv_dropout: Dropout probability
        batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_size: int = 3,
        nb_conv: int = 2,
        ndims: int = 3,
        padding: str = "same",
        dilation_rate: int = 1,
        activation: str = "elu",
        use_residuals: bool = False,
        conv_dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.nb_conv = nb_conv
        self.use_residuals = use_residuals
        self.conv_dropout = conv_dropout
        self.batch_norm = batch_norm
        self.ndims = ndims

        # Select convolution based on dimensions
        if ndims == 3:
            ConvLayer = nn.Conv3d
            self.dropout_noise_shape = (1, out_channels, 1, 1, 1)
        elif ndims == 2:
            ConvLayer = nn.Conv2d
            self.dropout_noise_shape = (1, out_channels, 1, 1)
        else:
            raise ValueError(f"ndims must be 2 or 3, got {ndims}")

        # Calculate padding for 'same' mode
        if padding == "same":
            pad = (conv_size - 1) // 2 * dilation_rate
        else:
            pad = 0

        # Build convolution layers
        self.convs = nn.ModuleList()
        for i in range(nb_conv):
            in_ch = in_channels if i == 0 else out_channels
            # Last conv in residual block has no activation (applied after residual add)
            self.convs.append(
                ConvLayer(
                    in_ch,
                    out_channels,
                    kernel_size=conv_size,
                    padding=pad,
                    dilation=dilation_rate,
                )
            )

        # Activation function
        self.activation = self._get_activation(activation)

        # Dropout
        if conv_dropout > 0:
            self.dropout = (
                nn.Dropout3d(conv_dropout) if ndims == 3 else nn.Dropout2d(conv_dropout)
            )
        else:
            self.dropout = None

        # Batch normalization
        if batch_norm:
            BatchNormLayer = nn.BatchNorm3d if ndims == 3 else nn.BatchNorm2d
            self.bn = BatchNormLayer(out_channels)
        else:
            self.bn = None

        # Residual connection adjustment
        if use_residuals and in_channels != out_channels:
            self.residual_conv = ConvLayer(
                in_channels, out_channels, kernel_size=conv_size, padding=pad
            )
        else:
            self.residual_conv = None

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == "elu":
            return nn.ELU(inplace=True)
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "leakyrelu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == "linear" or activation is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the convolutional block."""
        residual = x

        # Apply convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x)

            # Apply activation (except for last conv in residual block)
            if i < self.nb_conv - 1 or not self.use_residuals:
                x = self.activation(x)

            # Apply dropout
            if self.dropout is not None:
                x = self.dropout(x)

        # Residual connection
        if self.use_residuals:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
                if self.dropout is not None:
                    residual = self.dropout(residual)

            x = x + residual
            x = self.activation(x)

        # Batch normalization
        if self.bn is not None:
            x = self.bn(x)

        return x


class ConvEncoder(nn.Module):
    """
    Fully Convolutional Encoder with downsampling path.

    Args:
        nb_features: Base number of features
        input_channels: Number of input channels
        nb_levels: Number of encoder levels (number of downsamplings)
        conv_size: Kernel size for convolutions
        ndims: Number of spatial dimensions (2 or 3)
        feat_mult: Feature multiplier for each level
        pool_size: Pooling size
        padding: Padding mode
        dilation_rate_mult: Dilation rate multiplier per level
        activation: Activation function name
        use_residuals: Whether to use residual connections
        nb_conv_per_level: Number of convolutions per level
        conv_dropout: Dropout probability
        batch_norm: Whether to use batch normalization
        layer_nb_feats: Optional list specifying exact number of features per layer
    """

    def __init__(
        self,
        nb_features: int,
        input_channels: int,
        nb_levels: int,
        conv_size: int,
        ndims: int = 3,
        feat_mult: int = 2,
        pool_size: Union[int, Tuple[int, ...]] = 2,
        padding: str = "same",
        dilation_rate_mult: int = 1,
        activation: str = "elu",
        use_residuals: bool = False,
        nb_conv_per_level: int = 2,
        conv_dropout: float = 0.0,
        batch_norm: bool = False,
        layer_nb_feats: Optional[List[int]] = None,
    ):
        super().__init__()

        self.nb_levels = nb_levels
        self.ndims = ndims

        # Max pooling
        if ndims == 3:
            MaxPoolLayer = nn.MaxPool3d
        elif ndims == 2:
            MaxPoolLayer = nn.MaxPool2d
        else:
            raise ValueError(f"ndims must be 2 or 3, got {ndims}")

        if isinstance(pool_size, int):
            pool_size = (pool_size,) * ndims

        # Build encoder levels
        self.encoder_blocks = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        lfidx = 0
        current_channels = input_channels

        for level in range(nb_levels):
            # Calculate number of features for this level
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += nb_conv_per_level
            else:
                nb_lvl_feats = int(np.round(nb_features * (feat_mult**level)))

            # Dilation rate for this level
            dilation_rate = dilation_rate_mult**level

            # Create convolutional block
            block = ConvBlock(
                in_channels=current_channels,
                out_channels=nb_lvl_feats,
                conv_size=conv_size,
                nb_conv=nb_conv_per_level,
                ndims=ndims,
                padding=padding,
                dilation_rate=dilation_rate,
                activation=activation,
                use_residuals=use_residuals,
                conv_dropout=conv_dropout,
                batch_norm=batch_norm,
            )
            self.encoder_blocks.append(block)

            # Add max pooling (except for last level)
            if level < nb_levels - 1:
                self.pooling_layers.append(MaxPoolLayer(kernel_size=pool_size))

            current_channels = nb_lvl_feats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through encoder.

        Returns:
            Tuple of (final_output, skip_connections)
        """
        skip_connections = []

        for level in range(self.nb_levels):
            x = self.encoder_blocks[level](x)
            skip_connections.append(x)

            # Apply pooling (except for last level)
            if level < self.nb_levels - 1:
                x = self.pooling_layers[level](x)

        return x, skip_connections


class ConvDecoder(nn.Module):
    """
    Fully Convolutional Decoder with upsampling path and skip connections.

    Args:
        nb_features: Base number of features
        nb_levels: Number of decoder levels
        conv_size: Kernel size for convolutions
        nb_labels: Number of output channels
        ndims: Number of spatial dimensions (2 or 3)
        feat_mult: Feature multiplier for each level
        pool_size: Upsampling size (should match encoder pool_size)
        use_skip_connections: Whether to use skip connections (U-Net style)
        skip_n_concatenations: Number of top levels to skip concatenation
        padding: Padding mode
        dilation_rate_mult: Dilation rate multiplier per level
        activation: Activation function name
        use_residuals: Whether to use residual connections
        final_pred_activation: Final activation ('softmax', 'linear', etc.)
        nb_conv_per_level: Number of convolutions per level
        batch_norm: Whether to use batch normalization
        conv_dropout: Dropout probability
        layer_nb_feats: Optional list specifying exact number of features per layer
    """

    def __init__(
        self,
        nb_features: int,
        nb_levels: int,
        conv_size: int,
        nb_labels: int,
        ndims: int = 3,
        feat_mult: int = 2,
        pool_size: Union[int, Tuple[int, ...]] = 2,
        use_skip_connections: bool = True,
        skip_n_concatenations: int = 0,
        padding: str = "same",
        dilation_rate_mult: int = 1,
        activation: str = "elu",
        use_residuals: bool = False,
        final_pred_activation: str = "softmax",
        nb_conv_per_level: int = 2,
        batch_norm: bool = False,
        conv_dropout: float = 0.0,
        layer_nb_feats: Optional[List[int]] = None,
    ):
        super().__init__()

        self.nb_levels = nb_levels
        self.ndims = ndims
        self.use_skip_connections = use_skip_connections
        self.skip_n_concatenations = skip_n_concatenations
        self.final_pred_activation = final_pred_activation

        # Upsample layer
        if ndims == 3:
            self.upsample_mode = "trilinear"
        elif ndims == 2:
            self.upsample_mode = "bilinear"
        else:
            raise ValueError(f"ndims must be 2 or 3, got {ndims}")

        if isinstance(pool_size, int):
            self.pool_size = pool_size
        else:
            self.pool_size = pool_size[0]  # Assume uniform

        # Build decoder levels
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        lfidx = 0

        # First, calculate encoder features for skip connections
        encoder_features = []
        for enc_level in range(nb_levels):
            if layer_nb_feats is not None:
                enc_nb_feats = layer_nb_feats[enc_level * nb_conv_per_level]
            else:
                enc_nb_feats = int(np.round(nb_features * (feat_mult**enc_level)))
            encoder_features.append(enc_nb_feats)

        for level in range(nb_levels - 1):
            # Calculate number of features for this level
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[nb_levels * nb_conv_per_level + lfidx]
                lfidx += nb_conv_per_level
            else:
                nb_lvl_feats = int(
                    np.round(nb_features * (feat_mult ** (nb_levels - 2 - level)))
                )

            # Dilation rate for this level
            dilation_rate = dilation_rate_mult ** (nb_levels - 2 - level)

            # Calculate input channels (after upsampling and optional concatenation)
            # Input comes from upsampled features from previous decoder level
            if level == 0:
                # First decoder level: input is bottleneck (last encoder level)
                upsampled_channels = encoder_features[-1]
            else:
                # Subsequent levels: input is from previous decoder block
                upsampled_channels = int(
                    np.round(nb_features * (feat_mult ** (nb_levels - 1 - level)))
                )

            # Add skip connection channels if using skip connections
            in_channels = upsampled_channels
            if use_skip_connections:
                # Determine which encoder level to use for skip connection
                skip_idx = nb_levels - 2 - level
                if level < (nb_levels - skip_n_concatenations - 1):
                    # Concatenate with skip connection
                    in_channels = upsampled_channels + encoder_features[skip_idx]

            # Create convolutional block with correct input channels
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=nb_lvl_feats,
                conv_size=conv_size,
                nb_conv=nb_conv_per_level,
                ndims=ndims,
                padding=padding,
                dilation_rate=dilation_rate,
                activation=activation,
                use_residuals=use_residuals,
                conv_dropout=conv_dropout,
                batch_norm=batch_norm,
            )
            self.decoder_blocks.append(block)

        # Final prediction layer (1x1x1 convolution)
        if ndims == 3:
            self.final_conv = nn.Conv3d(nb_features, nb_labels, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(nb_features, nb_labels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x: Input tensor from encoder
            skip_connections: List of skip connection tensors from encoder

        Returns:
            Output prediction tensor
        """
        for level in range(self.nb_levels - 1):
            # Upsample
            x = F.interpolate(
                x,
                scale_factor=self.pool_size,
                mode=self.upsample_mode,
                align_corners=False if self.upsample_mode != "nearest" else None,
            )

            # Concatenate with skip connection
            if self.use_skip_connections and skip_connections is not None:
                if level < (self.nb_levels - self.skip_n_concatenations - 1):
                    skip_idx = self.nb_levels - 2 - level
                    skip = skip_connections[skip_idx]
                    x = torch.cat([skip, x], dim=1)

            # Apply convolutional block (input channels are now correct from initialization)
            x = self.decoder_blocks[level](x)

        # Final prediction layer
        x = self.final_conv(x)

        # Apply final activation
        if self.final_pred_activation == "softmax":
            x = F.softmax(x, dim=1)
        elif self.final_pred_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.final_pred_activation == "linear":
            pass  # No activation

        return x


class CustomUNet3D(nn.Module):
    """
    3D U-Net architecture for medical image segmentation and super-resolution.

    This is the main model used in SynthSR for synthesizing high-resolution MRI scans.

    Args:
        nb_features: Base number of features (default: 24)
        input_shape: Input shape as (C, D, H, W) for 3D or (C, H, W) for 2D
        nb_levels: Number of levels in the U-Net (default: 5)
        conv_size: Convolution kernel size (default: 3)
        nb_labels: Number of output channels (default: 1)
        feat_mult: Feature multiplier per level (default: 2)
        pool_size: Max pooling size (default: 2)
        padding: Padding mode (default: 'same')
        dilation_rate_mult: Dilation rate multiplier (default: 1)
        activation: Activation function (default: 'elu')
        skip_n_concatenations: Skip top N concatenations (default: 0)
        use_residuals: Use residual connections (default: False)
        final_pred_activation: Final activation ('softmax', 'linear', 'sigmoid')
        nb_conv_per_level: Number of convolutions per level (default: 2)
        conv_dropout: Dropout probability (default: 0.0)
        batch_norm: Use batch normalization (default: False)
        layer_nb_feats: Optional list of features per layer

    Example:
        >>> # Create a 3D UNet for single-channel MRI super-resolution
        >>> model = UNet3D(
        ...     nb_features=24,
        ...     input_shape=(1, 64, 64, 64),  # (C, D, H, W)
        ...     nb_levels=5,
        ...     conv_size=3,
        ...     nb_labels=1,
        ...     feat_mult=2,
        ...     final_pred_activation='linear'
        ... )
        >>>
        >>> # Forward pass
        >>> x = torch.randn(1, 1, 64, 64, 64)  # (B, C, D, H, W)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 1, 64, 64, 64])
    """

    def __init__(
        self,
        nb_features: int = 24,
        input_shape: Tuple[int, ...] = (1, 64, 64, 64),
        nb_levels: int = 5,
        conv_size: int = 3,
        nb_labels: int = 1,
        feat_mult: int = 2,
        pool_size: Union[int, Tuple[int, ...]] = 2,
        padding: str = "same",
        dilation_rate_mult: int = 1,
        activation: str = "elu",
        skip_n_concatenations: int = 0,
        use_residuals: bool = False,
        final_pred_activation: str = "linear",
        nb_conv_per_level: int = 2,
        conv_dropout: float = 0.0,
        batch_norm: bool = False,
        layer_nb_feats: Optional[List[int]] = None,
    ):
        super().__init__()

        # Determine ndims from input shape
        input_channels = input_shape[0]
        ndims = len(input_shape) - 1

        if ndims not in [2, 3]:
            raise ValueError(f"Only 2D and 3D UNets are supported, got {ndims}D")

        # Split layer_nb_feats for encoder and decoder
        enc_layer_nb_feats = None
        dec_layer_nb_feats = None
        if layer_nb_feats is not None:
            split_idx = nb_levels * nb_conv_per_level
            enc_layer_nb_feats = layer_nb_feats[:split_idx]
            dec_layer_nb_feats = layer_nb_feats[split_idx:]

        # Create encoder
        self.encoder = ConvEncoder(
            nb_features=nb_features,
            input_channels=input_channels,
            nb_levels=nb_levels,
            conv_size=conv_size,
            ndims=ndims,
            feat_mult=feat_mult,
            pool_size=pool_size,
            padding=padding,
            dilation_rate_mult=dilation_rate_mult,
            activation=activation,
            use_residuals=use_residuals,
            nb_conv_per_level=nb_conv_per_level,
            conv_dropout=conv_dropout,
            batch_norm=batch_norm,
            layer_nb_feats=enc_layer_nb_feats,
        )

        # Create decoder
        self.decoder = ConvDecoder(
            nb_features=nb_features,
            nb_levels=nb_levels,
            conv_size=conv_size,
            nb_labels=nb_labels,
            ndims=ndims,
            feat_mult=feat_mult,
            pool_size=pool_size,
            use_skip_connections=True,
            skip_n_concatenations=skip_n_concatenations,
            padding=padding,
            dilation_rate_mult=dilation_rate_mult,
            activation=activation,
            use_residuals=use_residuals,
            final_pred_activation=final_pred_activation,
            nb_conv_per_level=nb_conv_per_level,
            batch_norm=batch_norm,
            conv_dropout=conv_dropout,
            layer_nb_feats=dec_layer_nb_feats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D

        Returns:
            Output tensor of shape (B, nb_labels, D, H, W) for 3D or (B, nb_labels, H, W) for 2D
        """
        # Encoder forward pass
        x, skip_connections = self.encoder(x)

        # Decoder forward pass with skip connections
        x = self.decoder(x, skip_connections)

        return x


# ============================================================================
# MONAI Model Factory Functions
# ============================================================================


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
        # img_size=img_size,
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
        # pos_embed=pos_embed,
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
        # dropout_prob=dropout_prob,
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


def get_custom_unet3d(
    img_size: Tuple[int, int, int] = (128, 128, 128),
    in_channels: int = 1,
    out_channels: int = 1,
    nb_features: int = 24,
    nb_levels: int = 5,
    conv_size: int = 3,
    feat_mult: int = 2,
    use_residuals: bool = False,
    conv_dropout: float = 0.0,
    batch_norm: bool = False,
    final_pred_activation: str = "linear",
) -> nn.Module:
    """
    Custom UNet3D - Original SynthSR architecture.

    Good for: Baseline SynthSR model, proven architecture
    Note: This is the original architecture from the SynthSR paper

    Args:
        img_size: Input image size (D, H, W)
        in_channels: Number of input channels
        out_channels: Number of output channels
        nb_features: Base number of features (default: 24)
        nb_levels: Number of U-Net levels (default: 5)
        conv_size: Convolution kernel size (default: 3)
        feat_mult: Feature multiplier per level (default: 2)
        use_residuals: Use residual connections (default: False)
        conv_dropout: Dropout probability (default: 0.0)
        batch_norm: Use batch normalization (default: False)
        final_pred_activation: Final activation ('linear', 'sigmoid', 'softmax')
    """
    input_shape = (in_channels,) + img_size
    return CustomUNet3D(
        nb_features=nb_features,
        input_shape=input_shape,
        nb_levels=nb_levels,
        conv_size=conv_size,
        nb_labels=out_channels,
        feat_mult=feat_mult,
        use_residuals=use_residuals,
        conv_dropout=conv_dropout,
        batch_norm=batch_norm,
        final_pred_activation=final_pred_activation,
    )


# Model registry for easy access
MODEL_REGISTRY = {
    "custom_unet3d": get_custom_unet3d,  # Original SynthSR architecture
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
    # Custom UNet3D presets (original SynthSR architecture)
    "custom_unet3d_small": {
        "model_type": "custom_unet3d",
        "nb_features": 16,
        "nb_levels": 4,
        "use_residuals": False,
    },
    "custom_unet3d_base": {
        "model_type": "custom_unet3d",
        "nb_features": 24,
        "nb_levels": 5,
        "use_residuals": False,
    },
    "custom_unet3d_large": {
        "model_type": "custom_unet3d",
        "nb_features": 32,
        "nb_levels": 5,
        "use_residuals": True,
        "batch_norm": True,
    },

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
    if model_name in ["swinunetr", "unetr", "custom_unet3d"]:
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
