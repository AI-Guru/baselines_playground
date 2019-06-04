#!/bin/sh
brew install sdl sdl_image sdl_mixer sdl_ttf smpeg portmidi cmake
git clone https://github.com/garlicdevs/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
mkdir build && cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON -DMAC_OS=ON ..
make -j 4
cd ..
env MACOSX_DEPLOYMENT_TARGET=10.13 pip install .

pip install spiceypy pvl
