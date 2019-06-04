#!/bin/sh
apt-get update
pip install gym_ple
apt-get update && apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
pip install stable-baselines
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
pip install -e PyGame-Learning-Environment
pip install pygame
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
