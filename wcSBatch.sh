#!/bin/sh

#SBATCH --account=cms_svj
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --partition=gpu_gce
#SBATCH --gres=gpu:1
pwd
export dirname="/work1/cms_svj/keane/particleNet/SVJTaggerParticleNet/"
. "${dirname:?}/coffeaenvLCG/bin/activate" || exit
python train.py --outf logTest -C configs/C_tch_jConst.py
python validation.py --model net.pth --outf logTest -C logTest/config_out.py
exit
