# SVJ Tagger ParticleNet
The ParticleNet code was taken from https://github.com/colizz/weaver-benchmark.git
A description of how ParticleNet works can be found here at the CMS Machine Learning documentation: https://cms-ml.github.io/documentation/inference/particlenet.html

You can run this on the lpc-gpu, lxplus condor gpu, or the wilson cluster.

## Environment Setup
```
cd <working_directory>
git clone git@github.com:cms-svj/SVJTaggerNN.git
cd SVJTaggerNN
./setup.sh -l
source initLCG.sh
```
The `-l` flag tells the setup script to build the setup from LCG. Without the flag, the setup script will use a singularity container instead. Currently, the singularity container option is working on the lpc-gpu, but it does not work properly on the wilson cluster or the lxplus.
So before the singularity setup code was fixed, it is probably best to use the LCG setup code for now.

## Before Training
### Update Config File
There are three locations where you can access the input training files depending on where you are running the training.
This information is important for you to update the config file (`configs/C_tch_jConst.py`), so that you are able to access the input files. 
- lxplus: `/eos/user/c/chin/SVJTrainingFiles/jetConstTrainingFiles/`
- lpc gpu: `root://cmseos.fnal.gov//store/user/keanet/tchannel/jetConstTrainingFiles/jetConstTrainingFiles/`
- wilson cluster: `/work1/cms_svj/keane/particleNet/jetConstTrainingFiles/`
### Process Input Data
Since the input data for the particleNet training have to be in a certain shape in order to run the code successfully, we need to process the training root files to get the input data in the proper shape first. Once we have processed the root files, we will get an npz file in the `processedDataNPZ` folder.
```
python processData.py -C configs/C_tch_jConst.py
```
The training code takes the npz file as input and runs the training. 
The validation code also takes the same npz file as input, which significantly shortens the time to run these codes, since the input data only have to be processed once, instead of being processed before every training and before every validation.
However, notice that the npz file depends on the hyperparameter `numConst` in the config file (`configs/C_tch_jConst.py`).
`numConst` is the maximum number of constituents we keep for each jet in the training. The default is 100. If you vary this number, make sure you process the input data again to get another npz file, so that the input data will be in the desired shape.

## Training and Validation
```
python train.py --outf logTest -C configs/C_tch_jConst.py
python validation.py --model net.pth --outf logTest -C logTest/config_out.py
```

## Other Useful Information

### Which GPU to Use?
To know what GPU you are using, run the command below
`nvidia-smi -L`
The lpc-gpus are `P100`. They ran into memory problem (CUDA out of memory...) during training. 
On both the lxplus and wilson cluster, you can request for `V100` GPUs which have more memory. We used `V100` for our training. For some combinations of the hypermeter values, which make the network more complicated, we can still run into memory problem even when running on `V100`.

### Using wilson cluster
Once you have ssh-ed into wilson cluster, set up the environment and processed the input data.
After that, to run the training interactively, run
```
srun --account cms_svj --pty --constraint v100 --nodes=1 --partition gpu_gce --gres=gpu:1 bash
```
to get onto a `v100` GPU. See https://computing.fnal.gov/wilsoncluster/slurm-job-scheduler/ for information about the different flags.
Now run
```
source initLCG.sh
```
to get the environment. After that, you can run the training and validation using the commands in the `Training and Validation` section.

If you want to send a batch job instead of running interactively, you can run
```
sbatch wcSBatch.sh
```
This script assumes that you already set up the LCG environment before sending the batch job. Run `squeue` to monitor progress.

### Using lxplus condor gpu
Before doing anything, make sure you have set up the environment and processed the input data.
In the `lxplusCondor` directory, there are two files that allow us to submit job to the lxplus condor gpu.
The submit.sub is setup to request for `V100`. Once submitted, we are simply running the script `lxplusCondor/quickTestSetup.sh`. 
By default, `lxplusCondor/quickTestSetup.sh` is making a soft link to `/afs/cern.ch/user/c/chin/SVJTaggerParticleNet`.
You should open `lxplusCondor/quickTestSetup.sh` and change `/afs/cern.ch/user/c/chin/SVJTaggerParticleNet` to your working directory for SVJTaggerParticleNet on the lxplus!

Now, to run the training and validation on the lxplus condor gpu, once you are in the `lxplusCondor` directory, and run
```
condor_submit submit.sub
```
You can also add `-interactive` right after `condor_submit` for an interactive session, which is useful for troubleshooting.
Note: Making a soft link like that is a probably a bad idea. However, tarball for the entire LCG environment is huge (~1.5 GB). This is why I chose to do the former. 
For more information about using GPU on the lxplus condor, see https://batchdocs.web.cern.ch/tutorial/exercise10.html
### Hyperparameters
`configs/C_tch_jConst.py` contains the hyperparameters that we can vary for the training.
In the file, variables that start with `config.hyper` carry values for the hyperparameters.
Below is a description of some of the hyperparameters:
- numConst: The maximum number of jet constituents in each jet. Default = 100.
- num_of_k_nearest: The number of k nearest neightbors of each point. Default = 16.
- num_of_edgeConv_convLayers: The number of convolution layers inside each edge convolution layer. Default = 3.
- num_of_edgeConv_dim: The dimensions of the edge convolution layers. Default = [64, 128, 256]. This means there are 3 edge convolution layers. Assuming the num_of_edgeConv_convLayers = 3 and num_of_k_nearest = 16, then the first edge convolution layer has the dimension (16, (64, 64, 64)), the second layer has (16, (128, 128, 128)), and the third has (16, (256, 256, 256)). Simply remove/add a number from the list if you want to change the number of edge convolution layers.
- num_of_fc_layers: The number of fully connected layers at the end of the network.
- num_of_fc_nodes: The number of nodes in each fully connected layer.
- fc_dropout: The dropout rate of the fully connected layer.
Other hyperparameters that are more self-explanatory that can be varied are learning rate, batch size, and epochs.
