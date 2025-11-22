import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union


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


class UNet3D(nn.Module):
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
