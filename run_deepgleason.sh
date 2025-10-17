#!/bin/bash

# DeepGleason inference script
# Usage: ./run_deepgleason.sh <input_slide> <output_dir>

# Configuration variables
INPUT_SLIDE="${1:-}"
OUTPUT_DIR="${2:-./output}"
MODEL="${3:-models/model.ConvNeXtBase.hdf5}"
PREDICTIONS="${OUTPUT_DIR}/predictions.csv"
GENERATE_OVERLAY="${4:-false}"

# Check if input slide is provided
if [ -z "$INPUT_SLIDE" ]; then
    echo "Error: Input slide path is required"
    echo "Usage: $0 <input_slide> [output_dir] [model_path] [generate_overlay]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/slide.ome.tiff"
    echo "  $0 /path/to/slide.ome.tiff ./results"
    echo "  $0 /path/to/slide.ome.tiff ./results models/model.DenseNet121.hdf5"
    echo "  $0 /path/to/slide.ome.tiff ./results models/model.ConvNeXtBase.hdf5 true"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_SLIDE" ]; then
    echo "Error: Input slide '$INPUT_SLIDE' does not exist"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model file '$MODEL' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "DeepGleason Inference"
echo "========================================="
echo "Input slide:    $INPUT_SLIDE"
echo "Output dir:     $OUTPUT_DIR"
echo "Model:          $MODEL"
echo "Predictions:    $PREDICTIONS"
echo "Generate overlay: $GENERATE_OVERLAY"
echo "========================================="

# Build command
CMD="python code/main.py --input \"$INPUT_SLIDE\" --output \"$OUTPUT_DIR\" --model \"$MODEL\" --predictions \"$PREDICTIONS\""

# Add overlay flag if requested
if [ "$GENERATE_OVERLAY" = "true" ]; then
    CMD="$CMD --generate_overlay"
fi

# Run the command
echo "Running: $CMD"
echo ""
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "Error: Inference failed"
    echo "========================================="
    exit 1
fi
