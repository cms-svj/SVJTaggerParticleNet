import os
import numpy as np
from collections import OrderedDict
import itertools
import re
from glob import glob

allTargetConfigs = '''E_10_lr0p001_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_11_lr0p001_bs512_nc100_nk16_ed64_128_256_el2_fl7_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py
E_12_lr0p001_bs512_nc100_nk16_ed64_128_256_el2_fl7_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_13_lr0p001_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py
E_14_lr0p001_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_15_lr0p001_bs512_nc100_nk16_ed64_128_256_el3_fl7_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py
E_16_lr0p001_bs512_nc100_nk16_ed64_128_256_el3_fl7_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_1_lr0p001_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py
E_2_lr0p001_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_3_lr0p001_bs512_nc100_nk16_ed64_128_el2_fl7_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py
E_4_lr0p001_bs512_nc100_nk16_ed64_128_el2_fl7_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_5_lr0p001_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py
E_6_lr0p001_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_7_lr0p001_bs512_nc100_nk16_ed64_128_el3_fl7_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py
E_8_lr0p001_bs512_nc100_nk16_ed64_128_el3_fl7_fn256_fd0p3_ep120_dc0p0_rs30_4095_newSetupTTQCD.py
E_9_lr0p001_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep80_dc0p0_rs30_4095_newSetupTTQCD.py'''.splitlines()

for targetConfig in allTargetConfigs:
    #print("python train.py --outf logs/{} -C {}".format(targetConfig.replace("configs/batch/","").replace(".py",""),targetConfig))
    #os.system("python train.py --outf logs/{} -C {}".format(targetConfig.replace("configs/batch/","").replace(".py",""),targetConfig))
    # # # validation only
    print("python validation.py --model net.pth --outf logs/{0} -C logs/{0}/config_out.py".format(targetConfig.replace("configs/batch/","").replace(".py","")))
    os.system("python validation.py --model net.pth --outf logs/{0} -C logs/{0}/config_out.py".format(targetConfig.replace("configs/batch/","").replace(".py","")))
