#!/bin/bash

python3 tools/train.py -expn yolox_audio -n yolox_audio_x \
-f exps/yolox_audio/yolox_x.py -d 4 -b 32 --fp16 \
-c /data/pretrained/yolox_x.pth
