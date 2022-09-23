#!/bin/sh

#SBATCH --account=cms_svj
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --partition=gpu_gce
#SBATCH --gres=gpu:1

pwd
export dirname="/work1/cms_svj/keane/particleNet/SVJTaggerParticleNet/"
. "${dirname:?}/coffeaenvLCG/bin/activate" || exit
python batchRunSpecifyConfig.py
exit
