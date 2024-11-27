#!/bin/bash


for file in images/generated/*; do

  if [ -f "$file" ]; then
    filename=$(basename "$file" | cut -d. -f1)

    python3 src/image_mask.py  --input_image images/generated/"$filename".png \
     --output_mask images/generated/masks/"$filename"_mask.png
  fi  
done

python src/diff_protect.py attack.mode='none'

python src/diff_protect.py attack.mode='advdm' attack.g_mode='+'
python src/diff_protect.py attack.mode='advdm' attack.g_mode='-'

python src/diff_protect.py attack.mode='texture_only' attack.g_mode='+'
python src/diff_protect.py attack.mode='mist' attack.g_mode='+'

python src/diff_protect.py attack.mode='sds' attack.g_mode='+'
python src/diff_protect.py attack.mode='sds' attack.g_mode='-'
python src/diff_protect.py attack.mode='sds' attack.g_mode='-' attack.using_target=True