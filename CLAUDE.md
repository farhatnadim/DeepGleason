# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepGleason is a deep learning system for automated Gleason grading of prostate cancer using whole-slide images. It processes large histopathology images (whole-slide images) by tiling them into 1024x1024 patches, classifying each patch using a ConvNeXt or DenseNet model, and reconstructing the results into a color-coded overlay BigTIFF.

**Classification Categories (6 classes):**
- Artefact Sponge (A_S) - Gray (0.3, 0.3, 0.3)
- Artefact Empty/Unclear (A_D) - Black (0, 0, 0)
- Regular Tissue (R) - Green (0, 1, 0)
- Gleason 3 (G3) - Yellow (1, 1, 0)
- Gleason 4 (G4) - Orange (1, 0.5, 0)
- Gleason 5 (G5) - Red (1, 0, 0)

## Development Commands

### Installation
```bash
# Clone repository with LFS models
git clone https://github.com/frankkramer-lab/DeepGleason.git
cd DeepGleason/
git lfs pull

# Install dependencies
pip install -r requirements.txt
```

**Note:** PyVIPS requires LibVIPS system library. See [PyVIPS installation guide](https://github.com/libvips/pyvips).

### Running the Application
```bash
# Basic usage
python code/main.py \
  --input /path/to/slide.ome.tiff \
  --output /output/dir/ \
  --model models/model.ConvNeXtBase.hdf5 \
  --predictions /output/predictions.csv

# With overlay generation
python code/main.py -i input.tiff -o ./output/ \
  --model models/model.DenseNet121.hdf5 \
  --generate_overlay \
  -p predictions.csv

# Custom cache location (useful when system drive has limited space)
python code/main.py -i input.tiff -o ./output/ \
  --cache /path/to/large/disk/ \
  --model models/model.ConvNeXtBase.hdf5
```

### Testing
```bash
# Run all tests
python -m unittest discover tests/

# Run specific test modules
python -m unittest tests.test_model
python -m unittest tests.test_usage

# Run individual test
python -m unittest tests.test_model.DeepGleasonModels.test_model_ConvNeXtBase_load
```

### Docker
```bash
# Build Docker image
docker build -t deepgleason .

# Run container
docker run -v /path/to/data:/data --rm deepgleason

# Pull from registry
docker pull ghcr.io/frankkramer-lab/deepgleason
```

## Architecture & Pipeline

### Processing Pipeline (code/main.py)
1. **Slide Preparation** - Load whole-slide image with PyVIPS, extract RGB channels
2. **Tiling** (proc.gen_tiles) - Split into 1024x1024 tiles, save as PNGs to cache
3. **Preprocessing** - Apply stain normalization and padding via AUCMEDI subfunctions
4. **Inference** (model.run_aucmedi) - Run ConvNeXt/DenseNet model on tiles
5. **Reconstruction** (proc.class_reassemble) - Reassemble predictions into color-coded overlay
6. **Output** - Save BigTIFF with pyramid structure

### Key Components

**code/main.py** - Main orchestration script
- Manages CLI arguments and configuration
- Iterates through input slides
- Coordinates tile generation, inference, and reconstruction
- Handles caching and cleanup
- Supports resuming interrupted runs (checks for existing output files)

**code/proc.py** - Image processing utilities
- `gen_tiles()` - Splits WSI into 1024x1024 patches using PyVIPS, saves to cache
- `class_reassemble()` - Reconstructs predictions into RGB color map for visualization

**code/model.py** - AUCMEDI inference wrapper
- `run_aucmedi()` - Loads model, creates DataGenerator with subfunctions, runs predictions
- Uses stain normalization (Reinhard method) for color consistency
- Extracts architecture name from model filename (e.g., "model.ConvNeXtBase.hdf5" → "ConvNeXtBase")
- Returns DataFrame with soft labels and predicted class per tile

**code/stain_normalization.py** - Custom AUCMEDI Subfunction
- Implements Reinhard stain normalization using histolab
- Normalizes color variations from different scanners/lab procedures
- Uses fixed target image (stainnormalize_target.png) for reproducibility

### Data Flow
```
WSI Input → gen_tiles() → PNG tiles in cache
         ↓
    AUCMEDI Pipeline (DataGenerator + Subfunctions)
         ↓
    Stain Normalization → Padding → Model Inference
         ↓
    Predictions DataFrame (soft labels + class)
         ↓
    class_reassemble() → Color-coded overlay BigTIFF
```

### Model Architecture
- **Primary Model:** ConvNeXtBase (models/model.ConvNeXtBase.hdf5, ~1GB)
- **Alternative:** DenseNet121 (models/model.DenseNet121.hdf5, ~88MB)
- Models trained via AUCMEDI framework on TensorFlow/Keras
- Input: 1024x1024 RGB tiles after stain normalization
- Output: 6-class probability distribution

## Important Implementation Details

### PyVIPS Configuration
- `VIPS_CONCURRENCY` environment variable controls parallelism
- Set to "0" during tile generation for max performance (proc.py:24)
- Set to "1" during reconstruction to avoid memory issues (main.py:185)
- Large uncompressed intermediaries require significant disk space

### Caching & Resume Behavior
- Intermediate tiles stored in cache directory (default: system temp)
- Script skips tile generation if PNG already exists (proc.py:72)
- Skips entire slide if output BigTIFF already exists (main.py:153, 184, 219)
- Cleanup removes tile cache after processing each slide (main.py:222-224)

### File Naming Convention
- Tiles named as: `{slide_name}_{x_coord:06d}_{y_coord:06d}.png`
- Example: `slide123_001024_002048.png`
- **Critical:** Assumes all input slide filenames are unique to avoid overwrites

### GPU Configuration
- Uses `CUDA_VISIBLE_DEVICES` to select GPU (main.py:106)
- Default GPU ID: 0 (override with `-g` flag)

### Output Formats
- **Predictions CSV:** Soft labels (A_S, A_D, R, G3, G4, G5) + predicted class per tile
- **BigTIFF:** Pyramidal tiled TIFF with JPEG compression, color-coded overlay
- **Overlay mode** (`--generate_overlay`): Blends predictions (30%) with original image (70%)

## Testing Notes

- `tests/test_model.py` - Verifies model files exist and can be loaded
- `tests/test_usage.py` - End-to-end tests using Kaggle prostate cancer dataset
  - Downloads test WSI from Kaggle competition (requires public API token in test)
  - Tests basic inference, overlay generation, and multiple slide processing
  - Uses temporary directories for all I/O

## Dependencies & Framework

Built on **AUCMEDI** (v0.8.1) - an in-house medical image classification framework providing:
- High-level API for deep learning pipelines
- Extensive preprocessing via Subfunction system
- DataGenerator for efficient batch processing
- Support for 2D/3D architectures and ensemble methods

Other key dependencies:
- TensorFlow 2.13.1 (with GPU support in Docker)
- PyVIPS 2.2.1 (for efficient large image handling)
- histolab 0.6.0 (for stain normalization)
- pandas, numpy, scikit-learn (data processing)
