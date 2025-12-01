# SynthSR - Deep Learning for MRI Super-Resolution

A PyTorch implementation of resolution-agnostic MRI super-resolution using domain randomization and the MONAI framework.

## Overview

This project implements a learning-based approach for robust MRI super-resolution that can handle images acquired at any resolution and orientation. The model is trained using domain randomization to synthesize realistic low-resolution images from high-resolution scans, enabling it to generalize across diverse acquisition protocols.

**Key Features:**
- Resolution-agnostic super-resolution (handles arbitrary input resolutions)
- Domain randomization for robust generalization
- MONAI-based preprocessing pipeline
- Interactive parameter tuning with Streamlit
- Automatic checkpoint resumption
- Support for anisotropic acquisitions (e.g., thick-slice clinical scans)

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- MONAI 1.0+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd synthsr_monai

# Install dependencies
pip install torch torchvision
pip install monai[all]
pip install nibabel numpy scipy tqdm matplotlib streamlit
```

## Usage

### Training

#### Model Architectures

The project supports multiple model architectures through a unified registry:

**Custom UNet3D (Original SynthSR)**
```bash
python train.py \
    --model_architecture unet3d \
    --hr_image_dir /path/to/hr/images \
    --model_dir ./models \
    --epochs 100 \
    --batch_size 1 \
    --atlas_res 1.0 1.0 1.0 \
    --output_shape 128 128 128
```

**MONAI Models (SegResNet, SwinUNETR, etc.)**
```bash
# SegResNet (Recommended - Fast & Efficient)
python train.py \
    --model_architecture monai \
    --model_name segresnet_base \
    --hr_image_dir /path/to/hr/images \
    --model_dir ./models

# SwinUNETR (Transformer-based)
python train.py \
    --model_architecture monai \
    --model_name swinunetr_base \
    --hr_image_dir /path/to/hr/images \
    --model_dir ./models

# Other available models: unetr, vnet, attention_unet, etc.
```

**Available Model Presets:**
- `custom_unet3d_base` - Original SynthSR architecture
- `segresnet_base` - Efficient residual U-Net (Recommended)
- `swinunetr_base` - Transformer-based (High performance, more memory)
- `unetr_base` - Vision Transformer U-Net
- `unet_base` - Standard MONAI U-Net

**Key Parameters:**
- `--model_architecture`: `unet3d` (custom) or `monai` (MONAI models)
- `--model_name`: Specific model preset (for MONAI architecture)
- `--hr_image_dir`: Directory containing high-resolution NIfTI images
- `--model_dir`: Directory to save model checkpoints
- `--atlas_res`: Target resolution in mm (x, y, z)
- `--min_resolution`: Minimum simulated resolution (default: 1.0 1.0 1.0)
- `--max_res_aniso`: Maximum anisotropic resolution (default: 9.0 9.0 9.0)
- `--use_cache`: Enable MONAI CacheDataset for faster training

**Smart Checkpoint Resumption:**

The training automatically detects model architecture from checkpoints:

```bash
# Just specify the checkpoint - model type is auto-detected!
python train.py --checkpoint ./models/regression_unet_epoch_0050.pth

# Or let it auto-find the latest checkpoint:
python train.py --model_dir ./models  # Automatically resumes if checkpoint exists
```

⚠️ If you specify different `--model_architecture` than the checkpoint, it will:
1. Warn you about the mismatch
2. Override with checkpoint's configuration
3. Resume with the correct architecture

**Data Augmentation Control:**
```bash
# Disable specific augmentations
python train.py \
    --model_architecture monai \
    --model_name segresnet_base \
    --hr_image_dir /path/to/hr/images \
    --model_dir ./models \
    --no_randomise_res \
    --no_bias_field \
    --no_intensity_aug
```

**CSV-based Training:**
```bash
python train.py \
    --model_architecture monai \
    --model_name segresnet_base \
    --csv_file data.csv \
    --base_dir /path/to/data \
    --model_dir ./models
```

CSV format: `filename,split` where split is `train` or `val`.

### Inference

Run super-resolution on new images. The inference automatically detects model architecture from checkpoint:

```bash
# Single file - Works with ANY model type!
python test.py \
    --input /path/to/input.nii.gz \
    --output output_sr.nii.gz \
    --model ./models/regression_unet_final.pth \
    --atlas_res 1.0 1.0 1.0

# Batch processing (directory)
python test.py \
    --input /path/to/input/dir \
    --output /path/to/output/dir \
    --model ./models/regression_unet_final.pth \
    --tta  # Enable test-time augmentation
```

**Automatic Model Detection:**

The inference script automatically loads the correct architecture based on the checkpoint:
- ✅ CustomUNet3D checkpoints → Loads CustomUNet3D
- ✅ SegResNet checkpoints → Loads SegResNet
- ✅ SwinUNETR checkpoints → Loads SwinUNETR
- ✅ Any other MONAI model → Loads correctly

No need to specify model architecture during inference!

**Options:**
- `--tta`: Enable test-time augmentation (averaging with flipped predictions)
- `--no-tta`: Disable test-time augmentation
- `--save-intermediates`: Save intermediate volumes for debugging

### Interactive Parameter Tuning

Visualize and tune augmentation parameters in real-time:

```bash
streamlit run streamlit_lr_viewer.py
```

Features:
- Load and visualize NIfTI volumes
- Adjust all augmentation parameters interactively
- View original vs degraded comparisons
- Track intensity changes through pipeline
- Export generated low-resolution images

## Project Structure

```
synthSR/
├── train.py                    # Training script (supports all architectures)
├── train_diffusion.py          # Diffusion model training
├── test.py                     # Inference script (auto-detects architecture)
├── test_diffusion.py           # Diffusion inference
├── streamlit_lr_viewer.py      # Interactive visualization tool
├── src/
│   ├── models.py              # Unified model registry (Custom + MONAI)
│   ├── diff_models.py         # Diffusion models
│   ├── data_fft.py            # Data loading and LR-HR pair generation
│   ├── domain_rand.py         # Augmentation transforms
│   └── utils.py               # Utilities and checkpoint management
├── Dockerfile                  # Docker container definition
├── requirements.txt            # Python dependencies
└── README.md
```

## Domain Randomization Pipeline

The training pipeline simulates realistic low-resolution acquisitions through:

1. **Spatial Deformation**: Affine and elastic transformations
2. **Bias Field Corruption**: MRI intensity inhomogeneity simulation
3. **Intensity Augmentation**: Gamma correction and outlier clipping
4. **Resolution Randomization**: Simulated acquisition at varying resolutions
   - Anti-aliasing blur
   - Downsampling to target resolution
   - Gaussian noise injection
   - Upsampling to original shape

## Model Architectures

The project supports multiple model architectures through a unified interface:

### Custom UNet3D (Original SynthSR)
- **Network**: 3D U-Net with 5 levels
- **Features**: 24 base features (doubled at each level)
- **Input/Output**: Single-channel 3D volumes
- **Activation**: Linear (regression task)
- **Parameters**: ~1.2M

### MONAI Models

**SegResNet (Recommended)**
- **Network**: Residual encoder-decoder with VAE regularization
- **Features**: Efficient design with residual blocks
- **Best for**: Fast training, good performance-to-cost ratio
- **Parameters**: Variable based on configuration

**SwinUNETR**
- **Network**: Swin Transformer encoder + CNN decoder
- **Features**: Hierarchical vision transformer
- **Best for**: High performance, capturing long-range dependencies
- **Parameters**: ~62M (larger memory footprint)

**Other Available Models:**
- **UNETR**: Vision Transformer U-Net
- **Attention U-Net**: U-Net with attention gates
- **V-Net**: 3D architecture with residual connections
- **DynUNet**: Highly configurable dynamic U-Net

### Training Configuration
- **Loss**: L1 (Mean Absolute Error), SSIM, or combined
- **Optimizer**: Adam with Cosine LR schedule + warmup
- **Data**: Resolution-agnostic training with domain randomization
- **Augmentation**: Deformation, bias field, intensity, resolution randomization

## Citation

If you use this code, please cite the original SynthSR work:

```bibtex
@article{iglesias2021synthsr,
  title={Joint super-resolution and synthesis of 1 mm isotropic MP-RAGE volumes from clinical MRI exams with scans of different orientation, resolution and contrast},
  author={Iglesias, Juan Eugenio and Billot, Benjamin and Balbastre, Yael and Magdamo, Colin and Arnold, Steven E and Das, Sanjay and Edlow, Brian L and Alexander, Daniel C and Golland, Polina and Fischl, Bruce},
  journal={NeuroImage},
  volume={237},
  pages={118206},
  year={2021},
  publisher={Elsevier}
}
```

## License

Apache 2.0

---

For questions or issues, please open an issue on the repository.
