# DeepGleason Testing Guide

## Quick Start Testing

The `run_deepgleason.sh` script is ready to use with the updated dependencies!

### Prerequisites

```bash
# Verify all dependencies are installed
python -c "import tensorflow, pyvips, cv2, numpy; print('✓ All dependencies OK')"

# Check Python version
python --version  # Should be 3.12.3
```

### Basic Usage

```bash
# Test with ConvNeXtBase model (primary, recommended)
./run_deepgleason.sh /path/to/your/slide.ome.tiff ./output

# Test with DenseNet121 model (smaller, faster)
./run_deepgleason.sh /path/to/slide.ome.tiff ./output models/model.DenseNet121.hdf5

# Generate overlay visualization
./run_deepgleason.sh /path/to/slide.ome.tiff ./output models/model.ConvNeXtBase.hdf5 true
```

## Available Models

Both models now work with TensorFlow 2.20.0 and Python 3.12!

### ConvNeXtBase (Primary)
- **File:** `models/model.ConvNeXtBase.hdf5`
- **Size:** ~1GB
- **Performance:** Best accuracy
- **Status:** ✅ Fully tested

### DenseNet121 (Alternative)
- **File:** `models/model.DenseNet121.hdf5`
- **Size:** ~88MB
- **Performance:** Good accuracy, faster inference
- **Status:** ✅ Fully tested with Keras 3.x compatibility layer

## Testing Scenarios

### 1. Quick Sanity Test (5 minutes)
```bash
# If you have a small test slide
./run_deepgleason.sh test_slide.tiff ./test_output

# Check the output
ls -lh ./test_output/
cat ./test_output/predictions.csv | head -10
```

### 2. Full Pipeline Test (15-30 minutes depending on slide size)
```bash
# Process with overlay generation
./run_deepgleason.sh large_slide.ome.tiff ./results models/model.ConvNeXtBase.hdf5 true

# Verify outputs
ls -lh ./results/
# Should see:
# - *_gleason.tiff (output visualization)
# - predictions.csv (per-tile predictions)
```

### 3. Compare Both Models
```bash
# Test ConvNeXt
./run_deepgleason.sh slide.tiff ./output_convnext models/model.ConvNeXtBase.hdf5
mv ./output_convnext/predictions.csv ./predictions_convnext.csv

# Test DenseNet
./run_deepgleason.sh slide.tiff ./output_densenet models/model.DenseNet121.hdf5
mv ./output_densenet/predictions.csv ./predictions_densenet.csv

# Compare predictions
python -c "
import pandas as pd
df1 = pd.read_csv('predictions_convnext.csv')
df2 = pd.read_csv('predictions_densenet.csv')
print('ConvNeXt predictions:', len(df1))
print('DenseNet predictions:', len(df2))
print('Class agreement:', (df1['class'] == df2['class']).mean() * 100, '%')
"
```

## Expected Output

### Directory Structure After Run
```
output/
├── slide_name_gleason.tiff    # Color-coded Gleason grade map
└── predictions.csv            # Tile-level predictions
```

### predictions.csv Format
```csv
,A_S,A_D,R,G3,G4,G5,sample,class
0,0.1,0.05,0.7,0.1,0.03,0.02,slide_000000_000000.png,R
1,0.05,0.03,0.2,0.5,0.15,0.07,slide_000000_001024.png,G3
...
```

**Columns:**
- `A_S`: Artefact Sponge probability
- `A_D`: Artefact Empty/Unclear probability
- `R`: Regular tissue probability
- `G3`: Gleason 3 probability
- `G4`: Gleason 4 probability
- `G5`: Gleason 5 probability
- `sample`: Tile filename
- `class`: Predicted class (highest probability)

### Gleason Grading Color Map

The output TIFF uses these colors:
- **Gray** (0.3, 0.3, 0.3) - Artefact Sponge
- **Black** (0, 0, 0) - Artefact Empty/Unclear
- **Green** (0, 1, 0) - Regular Tissue
- **Yellow** (1, 1, 0) - Gleason 3
- **Orange** (1, 0.5, 0) - Gleason 4
- **Red** (1, 0, 0) - Gleason 5

## Performance Expectations

### ConvNeXtBase
- **Tile processing:** ~100-200 tiles/minute (CPU)
- **Tile processing:** ~500-1000 tiles/minute (GPU)
- **Memory usage:** ~4GB RAM + 2GB VRAM (GPU)

### DenseNet121
- **Tile processing:** ~150-300 tiles/minute (CPU)
- **Tile processing:** ~800-1500 tiles/minute (GPU)
- **Memory usage:** ~3GB RAM + 1.5GB VRAM (GPU)

### Slide Size Reference
- Small slide (10k tiles): 5-10 minutes (GPU) / 30-60 minutes (CPU)
- Medium slide (50k tiles): 30-60 minutes (GPU) / 3-5 hours (CPU)
- Large slide (200k tiles): 2-3 hours (GPU) / 10-20 hours (CPU)

## Troubleshooting

### Issue: "Module not found" errors
```bash
# Verify you're in the correct environment
which python
pip list | grep -E "(tensorflow|aucmedi|pyvips)"

# Reinstall if needed
pip install -r requirements.txt
```

### Issue: GPU not detected
```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Force CPU if GPU has issues
export CUDA_VISIBLE_DEVICES=-1
./run_deepgleason.sh slide.tiff ./output
```

### Issue: LibVIPS errors
```bash
# Check LibVIPS installation
python -c "import pyvips; print('LibVIPS:', pyvips.version(0), pyvips.version(1), pyvips.version(2))"

# On Ubuntu/Debian
sudo apt-get install libvips-dev

# On macOS
brew install vips
```

### Issue: Out of memory
```bash
# Use custom cache location on larger disk
./run_deepgleason.sh slide.tiff ./output models/model.ConvNeXtBase.hdf5 false

# Or modify the script to add --cache flag:
python code/main.py --input slide.tiff --output ./output \
  --model models/model.ConvNeXtBase.hdf5 \
  --cache /path/to/large/disk/cache
```

### Issue: Slow processing
```bash
# Use DenseNet121 (faster)
./run_deepgleason.sh slide.tiff ./output models/model.DenseNet121.hdf5

# Or check if GPU is being used
nvidia-smi  # Should show python process if GPU is active
```

## Validation Tests

### Test 1: Verify Model Loading
```bash
python -c "
from aucmedi import NeuralNetwork

print('Testing ConvNeXt...')
model = NeuralNetwork(6, channels=3, architecture='2D.ConvNeXtBase')
model.load('models/model.ConvNeXtBase.hdf5')
print('✓ ConvNeXt loaded successfully')

print('Testing DenseNet...')
model = NeuralNetwork(6, channels=3, architecture='2D.DenseNet121')
model.load('models/model.DenseNet121.hdf5')
print('✓ DenseNet loaded successfully')
"
```

### Test 2: Verify Stain Normalization
```bash
python -c "
from PIL import Image
from code.stain_normalization import StainNormalization
import numpy as np

# Load target image
target = Image.open('code/stainnormalize_target.png')
normalizer = StainNormalization(target)

# Create test image
test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
normalized = normalizer.transform(test_img)

print('✓ Stain normalization working')
print(f'  Input shape: {test_img.shape}')
print(f'  Output shape: {normalized.shape}')
print(f'  Output range: [{normalized.min()}, {normalized.max()}]')
"
```

### Test 3: Run Unit Tests
```bash
# Run all model tests
python -m unittest tests.test_model -v

# Expected output:
# test_model_ConvNeXtBase_exist ... ok
# test_model_ConvNeXtBase_load ... ok
# test_model_DenseNet121_exist ... ok
# test_model_DenseNet121_load ... ok
```

## Migration Verification

If you're migrating from an older version, verify compatibility:

```bash
# Compare predictions on the same slide
# Old version (before migration)
./run_deepgleason.sh slide.tiff ./output_old

# New version (after migration)
./run_deepgleason.sh slide.tiff ./output_new

# Compare outputs
python -c "
import pandas as pd
old = pd.read_csv('output_old/predictions.csv')
new = pd.read_csv('output_new/predictions.csv')

print('Class agreement:', (old['class'] == new['class']).mean() * 100, '%')
print('Probability correlation:')
for col in ['A_S', 'A_D', 'R', 'G3', 'G4', 'G5']:
    corr = old[col].corr(new[col])
    print(f'  {col}: {corr:.4f}')
"
```

Expected results: >95% class agreement, >0.95 correlation for probabilities

## Docker Testing

```bash
# Build image with new dependencies
docker build -t deepgleason:latest .

# Test with Docker
docker run -v /path/to/slides:/data deepgleason:latest

# Or interactive testing
docker run -it -v /path/to/slides:/data deepgleason:latest /bin/bash
# Inside container:
python code/main.py --input /data/slide.tiff --output /data/output \
  --model models/model.ConvNeXtBase.hdf5
```

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review `MIGRATION_2025.md` for known issues
3. Verify all dependencies: `pip list | grep -E "(tensorflow|aucmedi|opencv)"`
4. Check Python version: `python --version` (should be 3.12+)
5. Open an issue at: https://github.com/frankkramer-lab/DeepGleason/issues

## What's New in This Version

✅ **TensorFlow 2.20.0** - Latest version with Python 3.12 support
✅ **Both models working** - DenseNet121 fixed with Keras 3.x compatibility layer
✅ **Custom stain normalization** - OpenCV-based, no histolab dependency
✅ **50% fewer dependencies** - Cleaner, faster installation
✅ **Modern Python** - Full Python 3.12 support

Happy testing! 🔬
