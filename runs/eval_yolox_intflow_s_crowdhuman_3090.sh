#!/bin/bash

python3 tools/eval.py -expn yolox_oad_lm3__crowdhuman -n yolox_s_oad_lm3 \
-f exps/yolox_oad_lm3__crowdhuman/yolox_s_oad_lm3.py -d 4 -b 32 --fp16 \
-c YOLOX_outputs/yolox_oad_lm3__crowdhuman/best_ckpt.pth
