#!/bin/bash

INPUT_FOLDER="SHHQ-1.0_samples"
MASK_FOLDER="$INPUT_FOLDER/face_masks"

mkdir -p "$MASK_FOLDER"

for image_path in "$INPUT_FOLDER"/*.png; do
    filename=$(basename "$image_path")
    filename_no_ext="${filename%.*}"
    output_mask_path="$MASK_FOLDER/${filename_no_ext}_mask.png"

    python src/image_mask.py \
        --input_image "$image_path" \
        --output_mask "${output_mask_path}"

    echo "Processed $filename -> $output_mask_path"
done
