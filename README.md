# SVJTaggerNN

## Setup

```
cd <working_directory>
git clone git@github.com:cms-svj/SVJTaggerParticleNet.git
cd SVJTaggerParticleNet
./setup.sh
```

#Example
```
python train.py --outf logTest
python validation.py -C logTest/config_out.py --outf logTest

```

