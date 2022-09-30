import six

def make_schema(config_schema_dict):
    return [key+"."+val for key,vals in six.iteritems(config_schema_dict) for val in vals]

# define schema of config parameters
config_schema_dict = {
    "dataset":  ["path","signal","background","sample_fractions"],
    "features": ["uniform","weight","mT","jetConst","jetVariables","spectator"],
    "training": ["size","signal_id_method","signal_weight_method","weights","algorithms"],
    "hyper":    ["num_classes","learning_rate","batchSize","numConst","num_of_k_nearest","num_of_edgeConv_dim","num_of_edgeConv_convLayers","num_of_fc_layers","num_of_fc_nodes","fc_dropout","epochs","lambdaTag","lambdaReg","lambdaGR","lambdaDC","pTBins","n_pTBins","rseed","max_depth","n_estimators","subsample","min_samples_leaf","fl_coefficient","power","uniform_label","n_bins","uloss"],
}
config_schema = make_schema(config_schema_dict)

# include a default value and some required
config_defaults = {
    "dataset.path": None, "dataset.background": None, "dataset.signal": None
}
