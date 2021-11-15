python3 tools/demo.py --demo image -expn yolox_oad_lm3__crowdhuman -n yolox_s_oad_lm3 \
-f exps/yolox_oad_lm3__crowdhuman/yolox_s_oad_lm3.py \
-c YOLOX_outputs/yolox_oad_lm3__crowdhuman/best_ckpt.pth \
--path ./assets/crowdhuman1.jpg \
--save_folder /data/yolox_out \
--conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu