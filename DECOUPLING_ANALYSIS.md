# AUCMEDI Decoupling Analysis for DeepGleason

## Executive Summary

**Difficulty Level:** 🟡 **MODERATE** (3-5 days of work)

Decoupling AUCMEDI from DeepGleason is **feasible and worthwhile** for a future project. The coupling is relatively shallow - only 4 core functions are actually used from AUCMEDI's extensive framework.

## Current AUCMEDI Usage

### Actual Dependencies in DeepGleason Code

DeepGleason uses only **4 core components** from AUCMEDI:

1. **`input_interface`** (main.py:29, main.py:135)
   - Scans directory for image files
   - Returns file list and image format metadata

2. **`DataGenerator`** (model.py:26, model.py:59)
   - Batch loading of images
   - Applies preprocessing (subfunctions)
   - Feeds data to model

3. **`NeuralNetwork`** (model.py:26, model.py:47)
   - Wrapper around Keras model
   - Handles loading/prediction
   - Provides `meta_input` and `meta_standardize` attributes

4. **`Padding` subfunction** (model.py:27, model.py:41)
   - Single preprocessing step: square padding

5. **`Subfunction_Base`** (stain_normalization.py:27)
   - Base class for custom preprocessing (StainNormalization)

## What AUCMEDI Provides (But DeepGleason Doesn't Use)

AUCMEDI is a **comprehensive framework** with 137+ modules, but DeepGleason uses <5% of it:

### Unused Features
- ❌ AutoML capabilities
- ❌ Ensemble learning (bagging, stacking, boosting)
- ❌ Meta-learners (12+ algorithms)
- ❌ Built-in augmentation pipelines
- ❌ XAI/explainability methods (GradCAM, LIME, etc.)
- ❌ K-fold cross-validation
- ❌ Model evaluation utilities
- ❌ 50+ architecture implementations
- ❌ 3D/volume processing
- ❌ Multiple image loaders (DICOM, NIfTI, SimpleITK)

## Decoupling Strategy

### Phase 1: Abstract Core Dependencies (Low Risk)
**Estimated Time:** 1-2 days

Create thin wrapper interfaces:

```python
# interfaces/data_loader.py
class ImageDataLoader:
    """Abstract interface replacing DataGenerator"""
    def __init__(self, file_paths, image_dir, preprocessors, ...):
        pass

    def __len__(self): pass
    def __getitem__(self, idx): pass

# interfaces/model_wrapper.py
class ModelWrapper:
    """Abstract interface replacing NeuralNetwork"""
    def __init__(self, model_path):
        self.model = load_keras_model(model_path)
        self.input_shape = self.model.input_shape[1:3]

    def predict(self, data_loader): pass
```

### Phase 2: Implement Replacements (Medium Risk)
**Estimated Time:** 2-3 days

**2.1 Replace `input_interface`**
```python
# utils/file_scanner.py
def scan_image_directory(path):
    """Scan directory for supported image files"""
    extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    files = []
    for f in os.listdir(path):
        if Path(f).suffix.lower() in extensions:
            files.append(f)
    return sorted(files), detect_format(files[0])
```
**Complexity:** ⭐ Trivial

**2.2 Replace `DataGenerator`**
```python
# data/generator.py
class TileDataGenerator(tf.keras.utils.Sequence):
    """Replacement for AUCMEDI DataGenerator"""
    def __init__(self, file_list, image_dir, preprocessors, batch_size=32):
        self.files = file_list
        self.image_dir = image_dir
        self.preprocessors = preprocessors
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        for fname in batch_files:
            img = Image.open(os.path.join(self.image_dir, fname))
            img = np.array(img)
            # Apply preprocessors
            for preprocessor in self.preprocessors:
                img = preprocessor.transform(img)
            images.append(img)
        return np.array(images)
```
**Complexity:** ⭐⭐ Easy - Standard Keras pattern

**2.3 Replace `NeuralNetwork` wrapper**
```python
# model/predictor.py
class DeepGleasonPredictor:
    """Simplified model wrapper"""
    def __init__(self, model_path):
        self.model = load_model_keras3_compat(model_path)
        self.input_shape = self.model.input_shape[1:3]  # (224, 224)

    def predict(self, data_generator):
        return self.model.predict(data_generator, verbose=1)
```
**Complexity:** ⭐ Trivial

**2.4 Replace `Padding` subfunction**
```python
# preprocessing/padding.py
class SquarePadding:
    """Pads images to square aspect ratio"""
    def transform(self, image):
        h, w = image.shape[:2]
        if h == w:
            return image
        max_dim = max(h, w)
        padded = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)
        # Center the image
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
        return padded
```
**Complexity:** ⭐ Trivial

**2.5 Keep `Subfunction_Base`**
```python
# Already custom in stain_normalization.py
# Just remove the import, make it a local base class
class PreprocessorBase:
    """Base class for preprocessing steps"""
    def transform(self, image):
        raise NotImplementedError
```
**Complexity:** ⭐ Trivial

### Phase 3: Testing & Validation (Critical)
**Estimated Time:** 1 day

- Verify identical outputs on test tiles
- Benchmark performance (should be similar or faster)
- Test with both models (ConvNeXt, DenseNet)

## Benefits of Decoupling

### 1. Reduced Dependencies
**Before:** 24 packages (including AUCMEDI with ~15 sub-dependencies)
```
tensorflow, aucmedi, h5py, pyvips, opencv-python, pillow,
numpy, pandas, tqdm, scikit-learn, scikit-image, albumentations,
SimpleITK, batchgenerators, volumentations-aucmedi, plotnine,
pathos, matplotlib, keras-applications, classification-models-3D,
lime, pooch, kaggle
```

**After:** ~12 packages
```
tensorflow, h5py, pyvips, opencv-python, pillow, numpy,
pandas, tqdm, scikit-image (for resize), matplotlib (optional)
```

**Savings:** ~50% reduction in dependencies, ~500MB smaller installation

### 2. Faster Installation
- No AUCMEDI build time
- Fewer version conflicts
- Simpler Docker images

### 3. Better Control
- Custom preprocessing pipelines without AUCMEDI abstractions
- Direct TensorFlow/Keras usage (more maintainable)
- Easier debugging (no framework black boxes)

### 4. Future Flexibility
- Easier to switch to PyTorch if needed
- Can optimize for WSI-specific workflows
- Integration with other pathology frameworks (e.g., OpenSlide)

## Risks & Considerations

### Low Risk
- ✅ Core functionality is straightforward (data loading, prediction)
- ✅ AUCMEDI usage is shallow (no deep framework features)
- ✅ Stain normalization already decoupled

### Medium Risk
- ⚠️ **Standardization modes:** AUCMEDI has model-specific preprocessing
  - Need to extract/replicate for ConvNeXt/DenseNet
  - Documented in architecture files

- ⚠️ **Batch size tuning:** AUCMEDI auto-tunes batch sizes
  - Can hardcode reasonable defaults (32-64)

### Mitigation Strategy
1. **Keep AUCMEDI temporarily** during migration
2. **A/B test outputs** - Ensure predictions match exactly
3. **Document preprocessing** for each model architecture
4. **Phased rollout** - Replace one component at a time

## Recommended Approach

### Option A: Full Decoupling (Recommended for Production)
**Timeline:** 4-5 days
- Complete independence from AUCMEDI
- Minimal dependencies
- Full control over pipeline
- Best for long-term maintenance

### Option B: Partial Decoupling (Pragmatic)
**Timeline:** 2-3 days
- Keep AUCMEDI for convenience features
- Replace only DataGenerator + NeuralNetwork
- 70% of benefits with 50% effort

### Option C: Stay with AUCMEDI (Status Quo)
**Pros:**
- Zero migration effort
- Battle-tested framework
- Future AUCMEDI updates

**Cons:**
- Heavy dependency (20+ packages)
- Using <5% of framework
- Slower installation/deployment

## Code Change Summary

### Files to Modify
1. **`code/main.py`** (10 lines)
   - Replace `input_interface` → `scan_image_directory`

2. **`code/model.py`** (30 lines)
   - Replace `NeuralNetwork` → `DeepGleasonPredictor`
   - Replace `DataGenerator` → `TileDataGenerator`
   - Replace `Padding` → `SquarePadding`

3. **`code/stain_normalization.py`** (5 lines)
   - Replace `Subfunction_Base` → Local `PreprocessorBase`

### New Files to Create
1. `code/interfaces/data_loader.py` (~50 lines)
2. `code/interfaces/model_wrapper.py` (~30 lines)
3. `code/preprocessing/padding.py` (~20 lines)
4. `code/utils/file_scanner.py` (~30 lines)

**Total New Code:** ~130 lines (manageable)

## Conclusion

**Verdict: YES, decouple AUCMEDI in a future project**

### Why?
1. **High ROI:** 4-5 days of work for 50% fewer dependencies
2. **Low Risk:** Shallow integration makes migration straightforward
3. **Better Maintenance:** Direct TensorFlow usage is more transparent
4. **Performance:** May actually be faster without framework overhead

### When?
- **Now:** If planning significant future development
- **Later:** If current setup works and no issues

### How?
Start with **Option B (Partial Decoupling)**:
1. Replace DataGenerator first (2 days)
2. Test thoroughly with both models
3. Evaluate whether to continue with full decoupling

This approach minimizes risk while delivering most benefits.
