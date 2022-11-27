#!/bin/bash
#X11

#sudo docker login docker.io -u kmjeon -p # Type yourself

#Pull update docker image
sudo docker pull intflow/yolox_audio:dev_1.0_30xx_ubuntu18.04

#Run Dockers for YOLOXOAD
sudo docker run --name yolox_audio \
--gpus all --rm -p 6434:6434 \
--mount type=bind,src=/home/intflow/works,dst=/works \
--mount type=bind,src=/DL_data_big,dst=/data \
--mount type=bind,src=/DL_data,dst=/DL_data \
--mount type=bind,src=/NIA75_2022,dst=/NIA75_2022 \
--net=host \
--privileged \
--ipc=host \
-it intflow/yolox_audio:dev_1.0_30xx_ubuntu18.04 /bin/bash
