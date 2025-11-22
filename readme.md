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

Train a super-resolution model on high-resolution MRI scans:

```bash
python train.py \
    --hr_image_dir /path/to/hr/images \
    --model_dir ./models \
    --epochs 100 \
    --batch_size 1 \
    --atlas_res 1.0 1.0 1.0 \
    --output_shape 128 128 128
```

**Key Parameters:**
- `--hr_image_dir`: Directory containing high-resolution NIfTI images
- `--model_dir`: Directory to save model checkpoints
- `--atlas_res`: Target resolution in mm (x, y, z)
- `--min_resolution`: Minimum simulated resolution (default: 1.0 1.0 1.0)
- `--max_res_aniso`: Maximum anisotropic resolution (default: 9.0 9.0 9.0)
- `--use_cache`: Enable MONAI CacheDataset for faster training

**Data Augmentation Control:**
```bash
# Disable specific augmentations
python train.py \
    --hr_image_dir /path/to/hr/images \
    --model_dir ./models \
    --no_randomise_res \
    --no_bias_field \
    --no_intensity_aug
```

**CSV-based Training:**
```bash
python train.py \
    --csv_file data.csv \
    --base_dir /path/to/data \
    --model_dir ./models
```

CSV format: `filename,split` where split is `train` or `val`.

### Inference

Run super-resolution on new images:

```bash
# Single file
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
synthsr_monai/
├── train.py                    # Training script
├── test.py                     # Inference script
├── streamlit_lr_viewer.py      # Interactive visualization tool
├── src/
│   ├── model.py               # UNet3D architecture
│   ├── data.py                # Data loading and LR-HR pair generation
│   ├── domain_rand.py         # Augmentation transforms
│   └── utils.py               # Utilities and checkpoint management
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

## Model Architecture

- **Network**: 3D U-Net with 5 levels
- **Features**: 24 base features (doubled at each level)
- **Input/Output**: Single-channel 3D volumes
- **Activation**: Linear (regression task)
- **Loss**: L1 (Mean Absolute Error)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

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
