#!/bin/sh

#SBATCH --account=cms_svj
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --partition=gpu_gce
#SBATCH --qos=regular
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

pwd
export dirname="/work1/cms_svj/keane/particleNet/SVJTaggerParticleNet/"
. "${dirname:?}/coffeaenvLCG/bin/activate" || exit
python batchRunSpecifyConfig.py
exit
