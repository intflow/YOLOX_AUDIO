#!/bin/bash

#sudo docker login docker.io -u kmjeon -p  #Type yourself!
sudo docker commit yolox_audio yolox_audio:dev_1.0_30xx_ubuntu18.04
sudo docker tag yolox_audio:dev_1.0_30xx_ubuntu18.04 intflow/yolox_audio:dev_1.0_30xx_ubuntu18.04
sudo docker push intflow/yolox_audio:dev_1.0_30xx_ubuntu18.04
