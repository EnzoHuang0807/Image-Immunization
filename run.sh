#!/bin/bash

python src/diff_protect.py attack.mode='none'

python src/diff_protect.py attack.mode='advdm' attack.g_mode='+'
python src/diff_protect.py attack.mode='advdm' attack.g_mode='-'

python src/diff_protect.py attack.mode='texture_only' attack.g_mode='+'
python src/diff_protect.py attack.mode='mist' attack.g_mode='+'

python src/diff_protect.py attack.mode='sds' attack.g_mode='+'
python src/diff_protect.py attack.mode='sds' attack.g_mode='-'
python src/diff_protect.py attack.mode='sds' attack.g_mode='-' attack.using_target=True


# for i in $(seq -f "%03g" 1 30)
# do
#   input_image="${i}.png"
#   output_mask="${i}_mask.png"

#   python3 src/image_mask.py  --input_image "./images/generated/$input_image" \
#    --output_mask "./images/generated/masks/$output_mask"
  
# done