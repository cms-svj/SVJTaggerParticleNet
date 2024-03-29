import os
import numpy as np
from collections import OrderedDict
import itertools
import re
from glob import glob

allTargetConfigs = '''configs/batch/G_10_lr0p001_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_11_lr0p001_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_12_lr0p001_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_13_lr0p001_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_14_lr0p001_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_15_lr0p001_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_16_lr0p001_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_17_lr0p005_bs512_nc100_nk8_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_18_lr0p005_bs512_nc100_nk8_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_19_lr0p005_bs512_nc100_nk8_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_1_lr0p001_bs512_nc100_nk8_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_20_lr0p005_bs512_nc100_nk8_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_21_lr0p005_bs512_nc100_nk8_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_22_lr0p005_bs512_nc100_nk8_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_23_lr0p005_bs512_nc100_nk8_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_24_lr0p005_bs512_nc100_nk8_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_25_lr0p005_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_26_lr0p005_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_27_lr0p005_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_28_lr0p005_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_29_lr0p005_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_2_lr0p001_bs512_nc100_nk8_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_30_lr0p005_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_31_lr0p005_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_32_lr0p005_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_33_lr0p01_bs512_nc100_nk8_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_34_lr0p01_bs512_nc100_nk8_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_35_lr0p01_bs512_nc100_nk8_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_36_lr0p01_bs512_nc100_nk8_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_37_lr0p01_bs512_nc100_nk8_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_38_lr0p01_bs512_nc100_nk8_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_39_lr0p01_bs512_nc100_nk8_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_3_lr0p001_bs512_nc100_nk8_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_40_lr0p01_bs512_nc100_nk8_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_41_lr0p01_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_42_lr0p01_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_43_lr0p01_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_44_lr0p01_bs512_nc100_nk16_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_45_lr0p01_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_46_lr0p01_bs512_nc100_nk16_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_47_lr0p01_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_48_lr0p01_bs512_nc100_nk16_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_4_lr0p001_bs512_nc100_nk8_ed64_128_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_5_lr0p001_bs512_nc100_nk8_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_6_lr0p001_bs512_nc100_nk8_ed64_128_256_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_7_lr0p001_bs512_nc100_nk8_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py
configs/batch/G_8_lr0p001_bs512_nc100_nk8_ed64_128_256_el3_fl5_fn256_fd0p3_ep40_dc0p0_rs100_4095_QCD_TT_allDark.py
configs/batch/G_9_lr0p001_bs512_nc100_nk16_ed64_128_el2_fl5_fn256_fd0p3_ep40_dc0p0_rs30_4095_QCD_TT_allDark.py'''.splitlines()

for targetConfig in allTargetConfigs:
    print("python train.py --outf logs/{} -C {}".format(targetConfig.replace("configs/batch/","").replace(".py",""),targetConfig))
    os.system("python train.py --outf logs/{} -C {}".format(targetConfig.replace("configs/batch/","").replace(".py",""),targetConfig))
    # # # validation only
    print("python validation.py --model net_39.pth --outf logs/{0} -C logs/{0}/config_out.py".format(targetConfig.replace("configs/batch/","").replace(".py","")))
    os.system("python validation.py --model net_39.pth --outf logs/{0} -C logs/{0}/config_out.py".format(targetConfig.replace("configs/batch/","").replace(".py","")))