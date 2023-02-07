from magiconfig import MagiConfig

config = MagiConfig(dataset=MagiConfig(), features=MagiConfig(), training=MagiConfig(), hyper=MagiConfig())
config.dataset.path = "/work1/cms_svj/keane/trainingFiles/PNTrainingFiles/"
allsigs = [
"tree_SVJ_mMed-1000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-1500_mDark-20_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-2000_mDark-100_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-2000_mDark-1_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-2000_mDark-20_rinv-0p1_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-2000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-2000_mDark-20_rinv-0p5_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-2000_mDark-20_rinv-0p7_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-2000_mDark-50_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-3000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-4000_mDark-20_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-600_mDark-20_rinv-0p3_alpha-peak_yukawa-1_PN",
"tree_SVJ_mMed-800_mDark-20_rinv-0p3_alpha-peak_yukawa-1_PN",
]
config.dataset.signal = {"signal": list(sorted(set(allsigs)))}
config.dataset.background =  {	
								"QCD": 		[
												"tree_QCD_Pt_300to470_PN",
												"tree_QCD_Pt_470to600_PN",
												"tree_QCD_Pt_600to800_PN",
												"tree_QCD_Pt_800to1000_PN",
												"tree_QCD_Pt_1000to1400_PN",
												"tree_QCD_Pt_1400to1800_PN",
												"tree_QCD_Pt_1800to2400_PN",
												"tree_QCD_Pt_2400to3200_PN"
											],
								"TTJets": 	[
												"tree_TTJets_DiLept_genMET-150_PN",
												"tree_TTJets_DiLept_PN",
												"tree_TTJets_HT-1200to2500_PN",
												"tree_TTJets_HT-2500toInf_PN",
												"tree_TTJets_HT-600to800_PN",
												"tree_TTJets_HT-800to1200_PN",
												"tree_TTJets_Incl_PN",
												"tree_TTJets_SingleLeptFromTbar_genMET-150_PN",
												"tree_TTJets_SingleLeptFromTbar_PN",
												"tree_TTJets_SingleLeptFromT_genMET-150_PN",
												"tree_TTJets_SingleLeptFromT_PN"
											]
}
config.dataset.sample_fractions = [0.7, 0.15, 0.15]
jetConst = [
    'del_eta',
    'del_phi',
    'log_pt',
    'log_e',
    'log_pt_jetpt',
    'log_e_jete',
    'del_r',
]
# pdgId uses one-hot encoding
pdgIds = [-13,-211,1,11,13,130,2,211,22]
for pid in pdgIds:
    jetConst.append("pdgId{}".format(pid))
config.features.jetConst = jetConst
config.features.num_const = 100
config.features.jetVariables = [
"jCstPtAK8",
"jCstEtaAK8",
"jCstPhiAK8",
"jCstEnergyAK8",
"jCstAxismajorAK8",
"jCstAxisminorAK8",
"jCstTau1AK8",
"jCstTau2AK8",
"jCstTau3AK8",
"jCstPtDAK8",
"jCstSoftDropMassAK8"
]
# particleNet hyperparameters
config.hyper.learning_rate = 0.001
config.hyper.batchSize = 512
config.hyper.num_of_k_nearest = 16
config.hyper.num_of_edgeConv_dim = [64,128]
config.hyper.num_of_edgeConv_convLayers = 2
config.hyper.num_of_fc_layers = 5
config.hyper.num_of_fc_nodes = 256
config.hyper.fc_dropout = 0.3
config.hyper.epochs = 40
config.hyper.lambdaTag = 1.0
config.hyper.lambdaReg = 0.0
config.hyper.lambdaGR = 1.0 # keep this at 1 and change lambdaReg only
config.hyper.lambdaDC = 0.0
config.hyper.rseed = 100
config.hyper.num_classes = 3

