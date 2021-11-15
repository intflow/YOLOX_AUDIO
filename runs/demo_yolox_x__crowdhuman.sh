python3 tools/demo.py --demo video -expn yolox__crowdhuman -n yolox_x \
-f exps/yolox__crowdhuman/yolox_x.py \
-c /data/pretrained/yolox_x__crowdhuman.pth \
--path /data/AIGC_3rd_2021/set_01/set01_drone01.mp4 \
--save_folder /data/yolox_out \
--conf 0.15 --nms 0.65 --tsize 640 --save_result --device gpu