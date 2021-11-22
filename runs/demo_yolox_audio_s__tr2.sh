python3 tools/demo_audio.py --demo wav -expn yolox_audio__tr2 -n yolox_audio_s \
-f exps/yolox_audio__tr2/yolox_s.py \
-c /data/pretrained/yolox_s__AGC21_tr2.pth \
--path ./assets/test.wav \
--conf 0.2 --nms 0.65 --tsize_h 256 --tsize_w 512 --save_result --device gpu
