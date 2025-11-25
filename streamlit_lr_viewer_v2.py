"""
Streamlit Interactive Viewer for Frequency-Domain LR Creation Pipeline

This app demonstrates the FFT-based k-space cropping approach for creating
low-resolution MRI data, comparing it with the spatial-domain method.
"""

import streamlit as st
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
import os

try:
    from nilearn import plotting
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

from monai.transforms import GaussianSmooth, ScaleIntensityRangePercentiles
from src.data_fft import (
    HRLRDataGenerator,
    freq_domain_downsample_torch,
    FrequencyDomainDownsample,
)
from src.domain_rand import (
    RandomSpatialDeformation,
    BiasFieldCorruption,
    IntensityAugmentation,
)

# Page config
st.set_page_config(page_title="FFT LR Pipeline Viewer", layout="wide")

st.title("🔬 Frequency-Domain LR Creation Pipeline Viewer")
st.markdown("Interactively tune parameters and visualize FFT-based downsampling")

# Session state for caching
if "hr_image" not in st.session_state:
    st.session_state.hr_image = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None


@st.cache_data
def load_image(image_path):
    """Load NIfTI image and return as torch tensor"""
    nib_img = nib.load(str(image_path))
    img_data = nib_img.get_fdata()

    # Convert to tensor
    img_tensor = torch.from_numpy(img_data).float()
    # Add batch and channel dimensions: (1, 1, D, H, W)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    return img_tensor, nib_img.affine


def apply_pipeline_v2(hr_image, params):
    """
    Apply the frequency-domain LR creation pipeline.

    Returns both the transformed image and logs of min/max values at each step.
    """
    device = hr_image.device
    hr_image = hr_image.clone()
    lr_image = hr_image.clone()

    # Initialize logging
    logs = []
    logs.append(
        {
            "step": "0. Initial HR Image",
            "min": lr_image.min().item(),
            "max": lr_image.max().item(),
        }
    )

    # 1. Spatial Deformation
    if params["apply_deformation"]:
        deformer = RandomSpatialDeformation(
            scaling_bounds=params["scaling_bounds"]
            if params["scaling_bounds"] > 0
            else None,
            rotation_bounds=params["rotation_bounds"]
            if params["rotation_bounds"] > 0
            else None,
            shearing_bounds=params["shearing_bounds"]
            if params["shearing_bounds"] > 0
            else None,
            translation_bounds=params["translation_bounds"]
            if params["translation_bounds"] > 0
            else False,
            enable_90_rotations=params["enable_90_rotations"],
            elastic_sigma_range=(
                params["elastic_sigma_min"],
                params["elastic_sigma_max"],
            ),
            elastic_magnitude_range=(
                params["elastic_mag_min"],
                params["elastic_mag_max"],
            ),
            prob_deform=1.0,
        )
        lr_image = deformer(lr_image, interpolation="bilinear")
        logs.append(
            {
                "step": "1. After Spatial Deformation",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )
    else:
        logs.append(
            {
                "step": "1. Spatial Deformation (skipped)",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )

    # 2. Bias Field Corruption
    if params["apply_bias_field"]:
        bias = BiasFieldCorruption(
            bias_field_std=params["bias_field_std"],
            bias_scale=params["bias_scale"],
            prob=1.0,
        )
        lr_image = bias(lr_image)
        logs.append(
            {
                "step": "2. After Bias Field Corruption",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )
    else:
        logs.append(
            {
                "step": "2. Bias Field Corruption (skipped)",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )

    # 3. Intensity Augmentation
    if params["apply_intensity_aug"]:
        intensity_aug = IntensityAugmentation(
            clip=params["clip_value"] if params["clip_value"] > 0 else False,
            gamma_std=params["gamma_std"],
            channel_wise=False,
            prob_gamma=1.0 if params["gamma_std"] > 0 else 0.0,
        )
        lr_image = intensity_aug(lr_image)
        logs.append(
            {
                "step": "3. After Intensity Augmentation",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )

        # Apply fixed blur after intensity aug
        if params["apply_fixed_blur"]:
            fixed_blur = GaussianSmooth(sigma=params["fixed_blur_sigma"])
            # GaussianSmooth expects input without batch dimension (C, D, H, W)
            lr_image = fixed_blur(lr_image.squeeze(0)).unsqueeze(0)
            logs.append(
                {
                    "step": "3b. After Fixed Blur",
                    "min": lr_image.min().item(),
                    "max": lr_image.max().item(),
                }
            )
    else:
        logs.append(
            {
                "step": "3. Intensity Augmentation (skipped)",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )

    # 4. Normalize before FFT downsampling
    normalizer = ScaleIntensityRangePercentiles(
        lower=0, upper=100, b_min=0.0, b_max=1.0, clip=True
    )
    hr_image = normalizer(hr_image)  # Normalize HR too
    lr_image = normalizer(lr_image)
    logs.append(
        {
            "step": "4. After Normalization [0,1] (HR and LR both normalized)",
            "min": lr_image.min().item(),
            "max": lr_image.max().item(),
        }
    )

    # 5. Frequency-Domain Downsampling
    if params["apply_freq_downsampling"]:
        # Create resolution tensor
        resolution = torch.tensor(
            [[params["res_x"], params["res_y"], params["res_z"]]],
            dtype=torch.float32,
            device=device,
        )

        # Create frequency-domain downsampler
        freq_downsample = FrequencyDomainDownsample(
            volume_res=params["atlas_res"],
            target_res=params["target_res"],
            output_shape=list(hr_image.shape[2:]),
            noise_std=params["acquisition_noise_std"],
            prob_noise=1.0 if params["acquisition_noise_std"] > 0 else 0.0,
        )

        lr_image = freq_downsample(lr_image, resolution)
        logs.append(
            {
                "step": "5. After FFT Downsampling",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )
    else:
        logs.append(
            {
                "step": "5. FFT Downsampling (skipped)",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )

    # Final clipping to [0, 1]
    if params["clip_to_unit_range"]:
        # lr_image = torch.clamp(lr_image, 0.0, 1.0)
        lr_image = normalizer(lr_image)
        logs.append(
            {
                "step": "6. After Final Clipping [0,1]",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )
    else:
        logs.append(
            {
                "step": "6. Final Clipping (skipped)",
                "min": lr_image.min().item(),
                "max": lr_image.max().item(),
            }
        )

    return hr_image, lr_image, logs


# Sidebar - File Selection
st.sidebar.header("📁 Image Selection")
image_path = st.sidebar.text_input(
    "NIfTI Image Path", value="", placeholder="/path/to/your/image.nii.gz"
)

load_button = st.sidebar.button("Load Image")

if load_button and image_path:
    try:
        st.session_state.hr_image, st.session_state.affine = load_image(image_path)
        st.session_state.image_path = image_path
        st.sidebar.success(f"✅ Loaded image: {Path(image_path).name}")
        st.sidebar.info(f"Shape: {st.session_state.hr_image.shape}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading image: {str(e)}")

# Main interface
if st.session_state.hr_image is not None:
    hr_image = st.session_state.hr_image
    D, H, W = hr_image.shape[2:]

    # Info box explaining the difference
    st.info(
        """
    **🔬 Frequency-Domain Downsampling (FFT Method)**

    This approach uses k-space (frequency domain) cropping to simulate MRI acquisition:
    - More realistic simulation of MRI physics
    - Downsamples along the most anisotropic axis
    - Creates aliasing artifacts similar to real undersampled MRI
    - Used in the original downsampling paper
    """
    )

    # Create tabs for parameter groups
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔧 Deformation", "💡 Bias Field & Intensity", "🌊 Blur", "📐 FFT Downsampling"]
    )

    params = {}

    # Tab 1: Deformation Parameters
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            params["apply_deformation"] = st.checkbox("Enable Deformation", value=False)

            st.subheader("Affine Transformations")
            params["scaling_bounds"] = st.slider(
                "Scaling Bounds", 0.0, 0.5, 0.15, 0.01
            )
            params["rotation_bounds"] = st.slider(
                "Rotation Bounds (degrees)", 0.0, 45.0, 15.0, 1.0
            )
            params["shearing_bounds"] = st.slider(
                "Shearing Bounds", 0.0, 0.1, 0.012, 0.001
            )
            params["translation_bounds"] = st.slider(
                "Translation Bounds (voxels)", 0.0, 20.0, 10.0, 1.0
            )
            params["enable_90_rotations"] = st.checkbox(
                "Enable 90° Rotations", value=False
            )

        with col2:
            st.subheader("Elastic Deformation")
            params["elastic_sigma_min"] = st.slider(
                "Elastic Sigma Min", 0.0, 10.0, 5.0, 0.5
            )
            params["elastic_sigma_max"] = st.slider(
                "Elastic Sigma Max", 0.0, 15.0, 7.0, 0.5
            )
            params["elastic_mag_min"] = st.slider(
                "Elastic Magnitude Min", 0.0, 500.0, 100.0, 10.0
            )
            params["elastic_mag_max"] = st.slider(
                "Elastic Magnitude Max", 0.0, 500.0, 200.0, 10.0
            )

    # Tab 2: Bias Field & Intensity
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bias Field Corruption")
            params["apply_bias_field"] = st.checkbox("Enable Bias Field", value=False)
            params["bias_field_std"] = st.slider(
                "Bias Field Strength", 0.0, 1.0, 0.3, 0.05
            )
            params["bias_scale"] = st.slider(
                "Bias Field Smoothness", 0.01, 0.1, 0.025, 0.005
            )

        with col2:
            st.subheader("Intensity Augmentation")
            params["apply_intensity_aug"] = st.checkbox(
                "Enable Intensity Aug", value=False
            )
            params["clip_value"] = st.slider("Intensity Clipping", 0, 500, 300, 10)
            params["gamma_std"] = st.slider("Gamma Correction Std", 0.0, 1.0, 0.5, 0.05)

    # Tab 3: Blur
    with tab3:
        st.subheader("Fixed Gaussian Blur (MRI PSF Simulation)")
        params["apply_fixed_blur"] = st.checkbox("Enable Fixed Blur", value=True)
        params["fixed_blur_sigma"] = st.slider("Fixed Blur Sigma", 0.0, 2.0, 0.5, 0.1)

    # Tab 4: FFT Downsampling
    with tab4:
        col_top1, col_top2 = st.columns(2)

        with col_top1:
            params["apply_freq_downsampling"] = st.checkbox(
                "Enable FFT Downsampling", value=True
            )

        with col_top2:
            params["clip_to_unit_range"] = st.checkbox(
                "Clip to [0,1] Range", value=True
            )

        st.markdown("---")

        st.info(
            """
        **How FFT Downsampling Works:**
        1. Transforms image to frequency domain (k-space)
        2. Crops high frequencies (simulates undersampling)
        3. Transforms back to spatial domain
        4. Automatically selects the most anisotropic axis
        """
        )

        col1, col2, col3 = st.columns(3)

        params["atlas_res"] = [1.0, 1.0, 1.0]
        params["target_res"] = [1.0, 1.0, 1.0]

        with col1:
            st.subheader("Acquisition Resolution (mm)")
            params["res_x"] = st.slider("X Resolution", 1.0, 9.0, 1.0, 0.1)
            params["res_y"] = st.slider("Y Resolution", 1.0, 9.0, 1.0, 0.1)
            params["res_z"] = st.slider("Z Resolution", 1.0, 9.0, 4.0, 0.1)

        with col2:
            st.subheader("Acquisition Noise")
            params["acquisition_noise_std"] = st.slider(
                "Noise Std", 0.0, 0.1, 0.01, 0.001
            )

            # Calculate which axis will be downsampled
            factors = np.array([params["res_x"], params["res_y"], params["res_z"]])
            downsample_axis = np.argmax(factors)
            axis_names = ["X (Sagittal)", "Y (Coronal)", "Z (Axial)"]

            st.markdown("---")
            st.subheader("Downsampling Info")
            st.metric("Selected Axis", axis_names[downsample_axis])
            st.metric("Downsampling Factor", f"{factors[downsample_axis]:.1f}x")

        with col3:
            st.subheader("Geometry")
            is_isotropic = params["res_x"] == params["res_y"] == params["res_z"]
            st.metric("Type", "Isotropic" if is_isotropic else "Anisotropic")

            avg_res = (params["res_x"] + params["res_y"] + params["res_z"]) / 3
            quality = "High" if avg_res < 2.0 else "Medium" if avg_res < 4.0 else "Low"
            st.metric("Quality", quality)

    # Generate LR image
    st.markdown("---")

    generate_col1, generate_col2 = st.columns([3, 1])
    with generate_col1:
        st.subheader("🎯 Generated Low-Resolution Image (FFT Method)")
    with generate_col2:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.rerun()

    with st.spinner("Generating LR image with FFT downsampling..."):
        hr_image_norm, lr_image, intensity_logs = apply_pipeline_v2(hr_image, params)

    # Display intensity range logs
    st.markdown("### 📊 Intensity Range at Each Pipeline Step")
    log_data = []
    for log in intensity_logs:
        log_data.append(
            {
                "Pipeline Step": log["step"],
                "Min Value": f"{log['min']:.6f}",
                "Max Value": f"{log['max']:.6f}",
                "Range": f"{log['max'] - log['min']:.6f}",
            }
        )

    st.dataframe(log_data, use_container_width=True, hide_index=True)

    # Visualization
    st.markdown("---")

    # View options
    view_col1, view_col2, view_col3 = st.columns(3)

    with view_col1:
        view_axis = st.selectbox(
            "View Plane",
            options=["Axial (X-Y)", "Sagittal (Y-Z)", "Coronal (X-Z)"],
            index=0,
        )

    # Determine slice range based on view
    if "Axial" in view_axis:
        max_slice = D - 1
        default_slice = D // 2
    elif "Sagittal" in view_axis:
        max_slice = W - 1
        default_slice = W // 2
    else:  # Coronal
        max_slice = H - 1
        default_slice = H // 2

    with view_col2:
        slice_idx = st.slider("Slice", 0, max_slice, default_slice)

    with view_col3:
        colormap = st.selectbox(
            "Colormap", options=["gray", "viridis", "plasma", "hot"], index=0
        )

    # Extract slices
    hr_np = hr_image_norm.squeeze().cpu().numpy()
    lr_np = lr_image.squeeze().cpu().numpy()

    # Display with nilearn or matplotlib fallback
    if NILEARN_AVAILABLE:
        st.markdown("### 🔬 Medical Image Visualization (using nilearn)")

        # Create temporary NIfTI files for nilearn
        with tempfile.TemporaryDirectory() as tmpdir:
            hr_path = os.path.join(tmpdir, "hr.nii.gz")
            lr_path = os.path.join(tmpdir, "lr_fft.nii.gz")
            diff_path = os.path.join(tmpdir, "diff.nii.gz")

            # Save as NIfTI
            nib.save(nib.Nifti1Image(hr_np, st.session_state.affine), hr_path)
            nib.save(nib.Nifti1Image(lr_np, st.session_state.affine), lr_path)

            # Create orthogonal views
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Original HR - Orthogonal Views")
                fig1 = plt.figure(figsize=(12, 4))
                display = plotting.plot_anat(
                    hr_path,
                    display_mode='ortho',
                    cmap=colormap,
                    figure=fig1,
                    annotate=True,
                    draw_cross=True,
                    title="Original HR"
                )
                st.pyplot(fig1)
                plt.close()

            with col2:
                st.markdown("#### Generated LR (FFT) - Orthogonal Views")
                fig2 = plt.figure(figsize=(12, 4))
                display = plotting.plot_anat(
                    lr_path,
                    display_mode='ortho',
                    cmap=colormap,
                    figure=fig2,
                    annotate=True,
                    draw_cross=True,
                    title="Generated LR (FFT)"
                )
                st.pyplot(fig2)
                plt.close()

            # Show difference map
            st.markdown("---")
            st.markdown("#### 📊 Absolute Difference Map")
            diff_np = np.abs(hr_np - lr_np)
            nib.save(nib.Nifti1Image(diff_np, st.session_state.affine), diff_path)

            fig3 = plt.figure(figsize=(15, 4))
            display = plotting.plot_anat(
                diff_path,
                display_mode='ortho',
                cmap='hot',
                figure=fig3,
                annotate=True,
                draw_cross=True,
                title="Absolute Difference: |HR - LR|",
                colorbar=True
            )
            st.pyplot(fig3)
            plt.close()
    else:
        st.markdown("### 📊 Slice Views (matplotlib fallback)")
        st.info("💡 Install nilearn for better orthogonal visualization: `pip install nilearn`")

        if "Axial" in view_axis:
            hr_slice = hr_np[slice_idx, :, :]
            lr_slice = lr_np[slice_idx, :, :]
        elif "Sagittal" in view_axis:
            hr_slice = hr_np[:, :, slice_idx]
            lr_slice = lr_np[:, :, slice_idx]
        else:  # Coronal
            hr_slice = hr_np[:, slice_idx, :]
            lr_slice = lr_np[:, slice_idx, :]

        # Display side by side
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("#### Original HR")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            im1 = ax1.imshow(hr_slice.T, cmap=colormap, origin="lower")
            ax1.axis("off")
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            st.pyplot(fig1)
            plt.close()

        with col2:
            st.markdown("#### Generated LR (FFT)")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            im2 = ax2.imshow(lr_slice.T, cmap=colormap, origin="lower")
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2)
            plt.close()

        with col3:
            st.markdown("#### Absolute Difference")
            diff = np.abs(hr_slice - lr_slice)
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            im3 = ax3.imshow(diff.T, cmap="hot", origin="lower")
            ax3.axis("off")
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            st.pyplot(fig3)
            plt.close()

    # Statistics
    st.markdown("---")
    st.subheader("📊 Statistics")

    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

    with stat_col1:
        st.metric("HR Mean", f"{hr_np.mean():.4f}")
        st.metric("HR Std", f"{hr_np.std():.4f}")

    with stat_col2:
        st.metric("LR Mean", f"{lr_np.mean():.4f}")
        st.metric("LR Std", f"{lr_np.std():.4f}")

    with stat_col3:
        mae = np.abs(hr_np - lr_np).mean()
        st.metric("MAE", f"{mae:.4f}")

        mse = ((hr_np - lr_np) ** 2).mean()
        st.metric("MSE", f"{mse:.6f}")

    with stat_col4:
        rmse = np.sqrt(mse)
        st.metric("RMSE", f"{rmse:.4f}")

        # R²
        ss_res = np.sum((hr_np - lr_np) ** 2)
        ss_tot = np.sum((hr_np - hr_np.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        st.metric("R²", f"{r2:.4f}")

    with stat_col5:
        # PSNR
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
            st.metric("PSNR (dB)", f"{psnr:.2f}")
        else:
            st.metric("PSNR (dB)", "∞")

    # Display acquisition parameters used
    if params["apply_freq_downsampling"]:
        st.markdown("---")
        st.subheader("🎯 FFT Downsampling Parameters Used")

        acq_col1, acq_col2 = st.columns(2)

        with acq_col1:
            st.markdown("**Target Resolution (mm)**")
            st.code(
                f"X: {params['res_x']:.2f} mm\n"
                f"Y: {params['res_y']:.2f} mm\n"
                f"Z: {params['res_z']:.2f} mm"
            )

        with acq_col2:
            st.markdown("**Downsampling Method**")
            st.code("FFT k-space cropping\nAutomatic axis selection")

    # Save option
    st.markdown("---")
    save_col1, save_col2 = st.columns([3, 1])

    with save_col1:
        save_path = st.text_input(
            "Save LR Image Path", value="lr_output_fft.nii.gz", placeholder="output.nii.gz"
        )

    with save_col2:
        if st.button("💾 Save LR Image", use_container_width=True):
            try:
                lr_nib = nib.Nifti1Image(lr_np, st.session_state.affine)
                nib.save(lr_nib, save_path)
                st.success(f"✅ Saved to {save_path}")
            except Exception as e:
                st.error(f"❌ Error saving: {str(e)}")

else:
    # Landing page
    st.info("👈 Please load a NIfTI image from the sidebar to begin")

    st.markdown(
        """
    ### 🎯 What is this tool?

    This interactive viewer demonstrates **frequency-domain (FFT) downsampling** for creating
    low-resolution MRI data - an alternative to spatial-domain methods.

    ### 🔬 Why FFT Downsampling?

    **Frequency-domain downsampling**:
    - ✅ Simulates actual MRI k-space undersampling
    - ✅ Creates realistic aliasing artifacts
    - ✅ Better matches MRI acquisition physics
    - ✅ Used in the original downsampling paper

    **Spatial-domain downsampling** (traditional):
    - Uses interpolation (trilinear, etc.)
    - Smoother but less realistic
    - Doesn't match MRI physics as closely

    ### 📋 Pipeline Components

    1. **Spatial Deformation** - Optional anatomical variations
    2. **Bias Field Corruption** - MRI B1 field inhomogeneity
    3. **Intensity Augmentation** - Contrast variations
    4. **Fixed Gaussian Blur** - MRI point spread function
    5. **FFT Downsampling** - K-space cropping along most anisotropic axis
    6. **Noise Injection** - Scanner noise simulation

    ### 🚀 Getting Started

    1. Enter path to a NIfTI file in the sidebar
    2. Click "Load Image"
    3. Navigate through tabs to adjust parameters
    4. Enable FFT downsampling in the "FFT Downsampling" tab
    5. View results in real-time
    6. Compare with spatial-domain method using the original viewer

    ### 💡 Key Differences from Spatial Method

    | Feature | Spatial Method | FFT Method |
    |---------|----------------|------------|
    | Domain | Spatial | Frequency (k-space) |
    | Operation | Interpolation | FFT + crop + iFFT |
    | Artifacts | Blurring | Aliasing |
    | MRI Physics | Approximation | More realistic |
    | Axis | Multi-axis | Single (most anisotropic) |

    ### 📊 Recommended Settings

    **Typical Clinical Anisotropy** (4x in Z):
    - Resolution: 1×1×4 mm
    - Noise: 0.01
    - Disable other augmentations for testing

    **Heavy Anisotropy** (challenging):
    - Resolution: 1×1×6-9 mm
    - Enable bias field: 0.3 std
    - Enable gamma: 0.5 std
    - Noise: 0.02
    """
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "SynthSR with FFT Downsampling - Frequency-Domain LR Pipeline Viewer"
    "</div>",
    unsafe_allow_html=True,
)
