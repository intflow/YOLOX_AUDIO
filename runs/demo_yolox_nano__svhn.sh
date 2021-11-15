python3 tools/demo.py --demo image -expn yolox__svhn -n yolox_nano \
-f exps/yolox__svhn/yolox_nano.py \
-c YOLOX_outputs/yolox__svhn/best_ckpt.pth \
--path ./assets/tag4.jpg \
##--save_folder /data/yolox_out \
--conf 0.85 --nms 0.45 --tsize 320 --save_result --device gpu