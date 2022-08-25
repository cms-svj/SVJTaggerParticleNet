#!/bin/bash

ln -s /afs/cern.ch/user/c/chin/SVJTaggerParticleNet
cd SVJTaggerParticleNet
source initLCG.sh
python train.py --outf logTest -C configs/C_tch_jConst.py
python validation.py --model net.pth --outf logTest -C logTest/config_out.py
