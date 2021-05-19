from magiconfig import MagiConfig

config = MagiConfig(dataset=MagiConfig(), features=MagiConfig(), training=MagiConfig(), hyper=MagiConfig())
config.dataset.path = ""
config.dataset.signal = {"signal": ["tree_SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-peak_MC2017"]}
config.dataset.background =  {"background": [
    "tree_QCD_Pt_600to800_MC2017",
]}
config.features.uniform = ["pt"]
config.features.train = ["girth","tau21","tau32","msd","deltaphi","axisminor","axismajor","ptD","ecfN2b1","ecfN3b1","fChHad","fEle","fMu","fNeuHad","fPho"]
config.features.spectator = ["mt","eta"]
config.training.size = 0.5
config.training.signal_id_method = "two"
config.training.signal_weight_method = "default"
config.training.weights = {
    "flat": ["flatweightZ30"],
    "proc": ["procweight"],
}
config.training.algorithms = {
    "bdt": "flat",
    "ubdt": "proc",
}
config.hyper.max_depth = 3
config.hyper.n_estimators = 1000
config.hyper.subsample = 0.6
config.hyper.learning_rate = 1e-3
config.hyper.min_samples_leaf = 0.05
config.hyper.fl_coefficient = 3
config.hyper.power = 1.3
config.hyper.uniform_label = 1
config.hyper.n_bins = 20
config.hyper.uloss = "exp"
config.hyper.num_of_layers = 4
config.hyper.num_of_nodes = 40
config.hyper.dropout = 0.3
config.hyper.epochs = 1000
