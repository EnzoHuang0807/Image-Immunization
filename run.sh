#!/bin/bash


# for file in SHHQ-1.0_samples/*; do

#   if [ -f "$file" ]; then
#     filename=$(basename "$file" | cut -d. -f1)

#     python3 src/image_mask.py  --input_image ./SHHQ-1.0_samples/"$filename".png \
#      --output_mask ./SHHQ-1.0_samples/face_masks/"$filename"_mask.png
#   fi  
# done
CUDA_VISIBLE_DEVICES=5

# python src/diff_protect.py attack.mode='none'

# python src/diff_protect.py attack.mode='advdm' attack.g_mode='+'
# python src/diff_protect.py attack.mode='advdm' attack.g_mode='-'

# python src/diff_protect.py attack.mode='texture_only' attack.g_mode='-'
python src/diff_protect.py attack.mode='mist' attack.g_mode='+'

# python src/diff_protect.py attack.mode='sds' attack.g_mode='+'
# python src/diff_protect.py attack.mode='sds' attack.g_mode='-'
# python src/diff_protect.py attack.mode='sds' attack.g_mode='-' attack.using_target=True