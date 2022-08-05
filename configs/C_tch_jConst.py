from magiconfig import MagiConfig

config = MagiConfig(dataset=MagiConfig(), features=MagiConfig(), training=MagiConfig(), hyper=MagiConfig())
config.dataset.path = "/eos/user/c/chin/SVJTrainingFiles/jetConstTrainingFiles/"
allsigs = [
"tree_SVJ_mMed-1000_mDark-20_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-1500_mDark-20_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-2000_mDark-100_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-2000_mDark-1_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-2000_mDark-20_rinv-0.1_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-2000_mDark-20_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-2000_mDark-20_rinv-0.5_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-2000_mDark-20_rinv-0.7_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-2000_mDark-50_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-3000_mDark-20_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-4000_mDark-20_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-600_mDark-20_rinv-0.3_alpha-peak_yukawa-1_MC2018",
"tree_SVJ_mMed-800_mDark-20_rinv-0.3_alpha-peak_yukawa-1_MC2018"
]
config.dataset.signal = {"signal": list(sorted(set(allsigs)))}
config.dataset.background =  {"background": [
"tree_QCD_Pt_170to300_MC2018",
"tree_QCD_Pt_300to470_MC2018",
"tree_QCD_Pt_470to600_MC2018",
"tree_QCD_Pt_600to800_MC2018",
"tree_QCD_Pt_800to1000_MC2018",
"tree_QCD_Pt_1000to1400_MC2018",
"tree_QCD_Pt_1400to1800_MC2018",
"tree_QCD_Pt_1800to2400_MC2018",
"tree_QCD_Pt_2400to3200_MC2018",
"tree_QCD_Pt_3200toInf_MC2018",
]}
config.dataset.sample_fractions = [0.70, 0.15, 0.15]
config.features.uniform = "jCstPtAK8"
config.features.weight = "jCstWeightAK8"
config.features.mT = "jCstPtAK8"
config.features.train = [
"jCstPt",
"jCstEta",
"jCstPhi",
"jCstEnergy",
"jCstPdgId",
#"jCstAxismajorAK8",
#"jCstAxisminorAK8",
#"jCstdoubleBDiscriminatorAK8",
#"jCstTau1AK8",
#"jCstTau2AK8",
#"jCstTau3AK8",
#"jCstNumBhadronsAK8",
#"jCstNumChadronsAK8",
#"jCstPtDAK8",
#"jCstSoftDropMassAK8",
"jCsthvCategory",
"jCstEvtNum",
"jCstJNum"
]
config.hyper.learning_rate = 0.001
config.hyper.batchSize = 1024
config.hyper.num_of_layers_features = 2
config.hyper.num_of_layers_tag = 2
config.hyper.num_of_layers_pT = 5
config.hyper.num_of_nodes = 40
config.hyper.dropout = 0.3
config.hyper.epochs = 50
config.hyper.lambdaTag = 1.0
config.hyper.lambdaReg = 0.0
config.hyper.lambdaGR = 1.0 # keep this at 1 and change lambdaReg only
config.hyper.lambdaDC = 0.0
config.hyper.pTBins = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 4000, 4500]
config.hyper.n_pTBins = len(config.hyper.pTBins)
config.hyper.rseed = 30
