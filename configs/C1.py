from magiconfig import MagiConfig

config = MagiConfig(dataset=MagiConfig(), features=MagiConfig(), training=MagiConfig(), hyper=MagiConfig())
config.dataset.path = "root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/Run2ProductionV17/Skims/tree_dijetmtdetahadloosemf-train-flatsig/"
allsigs = [ "tree_SVJ_mZprime-"+str(mZprime)+"_mDark-20_rinv-0.3_alpha-peak_MC2017" for mZprime in range(1500,4600,100) ] \
        + [ "tree_SVJ_mZprime-3000_mDark-"+str(mDark)+"_rinv-0.3_alpha-peak_MC2017" for mDark in range(10,110,10) ] \
        + [ "tree_SVJ_mZprime-3000_mDark-20_rinv-"+str(rinv)+"_alpha-peak_MC2017" for rinv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] ] \
        + [ "tree_SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-"+alpha+"_MC2017" for alpha in ["peak","high","low"] ]
config.dataset.signal = {"signal": list(sorted(set(allsigs)))}
config.dataset.background =  {"background": [
    "tree_QCD_Pt_300to470_MC2017",
    "tree_QCD_Pt_470to600_MC2017",
    "tree_QCD_Pt_600to800_MC2017",
    "tree_QCD_Pt_800to1000_MC2017",
    "tree_QCD_Pt_1000to1400_MC2017",
    "tree_QCD_Pt_1400to1800_MC2017",
    "tree_QCD_Pt_1800to2400_MC2017",
    "tree_QCD_Pt_2400to3200_MC2017",
    "tree_TTJets_MC2017",
    "tree_TTJets_DiLept_MC2017",
    "tree_TTJets_DiLept_genMET150_MC2017",
    "tree_TTJets_SingleLeptFromT_MC2017",
    "tree_TTJets_SingleLeptFromT_genMET150_MC2017",
    "tree_TTJets_SingleLeptFromTbar_MC2017",
    "tree_TTJets_SingleLeptFromTbar_genMET150_MC2017",
    "tree_TTJets_HT600to800_MC2017",
    "tree_TTJets_HT800to1200_MC2017",
    "tree_TTJets_HT1200to2500_MC2017",
    "tree_TTJets_HT2500toInf_MC2017",
]}
config.dataset.sample_fractions = [0.70, 0.15, 0.15]
config.features.uniform = "pt"
config.features.train = ["girth","tau21","tau32","msd","deltaphi","axisminor","axismajor","ptD","ecfN2b1","ecfN3b1","fChHad","fEle","fMu","fNeuHad","fPho"]
config.hyper.learning_rate = 1e-3
config.hyper.batchSize = 5000
config.hyper.num_of_layers = 1
config.hyper.num_of_nodes = 40
config.hyper.dropout = 0.3
config.hyper.epochs = 10
config.hyper.lambdaTag = 1.0
config.hyper.lambdaReg = 1e-2
config.hyper.lambdaGR = 1.0
