#!/bin/bash

python src/diff_protect.py attack.mode='none'

python src/diff_protect.py attack.mode='advdm' attack.g_mode='+'
python src/diff_protect.py attack.mode='advdm' attack.g_mode='-'

python src/diff_protect.py attack.mode='texture_only' attack.g_mode='+'
python src/diff_protect.py attack.mode='mist' attack.g_mode='+'

python src/diff_protect.py attack.mode='sds' attack.g_mode='+'
python src/diff_protect.py attack.mode='sds' attack.g_mode='-'
python src/diff_protect.py attack.mode='sds' attack.g_mode='-' attack.using_target=True