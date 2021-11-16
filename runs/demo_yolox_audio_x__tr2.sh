python3 tools/demo_audio.py --demo image -expn yolox_audio__tr2 -n yolox_audio_x \
-f exps/yolox_audio__tr2/yolox_x.py \
-c YOLOX_outputs/yolox_audio__tr2/best_ckpt.pth \
--path /data/AIGC_3rd_2021/GIST_tr2_100/img/ \
--save_folder /data/yolox_out \
--conf 0.2 --nms 0.65 --tsize 256 --save_result --device gpu
