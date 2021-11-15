#!/bin/bash

python3 tools/train.py -expn yolox__svhn -n yolox_nano \
-f exps/yolox__svhn/yolox_nano.py -d 4 -b 32 --fp16 \
-c /data/pretrained/yolox_nano.pth
