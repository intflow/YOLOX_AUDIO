#!/bin/bash

python3 tools/train.py -expn yolox__crowdhuman -n yolox_x \
-f exps/yolox__crowdhuman/yolox_x.py -d 4 -b 32 --fp16 \
-c /data/pretrained/yolox_x.pth
