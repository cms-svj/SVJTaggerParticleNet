import six

def make_schema(config_schema_dict):
    return [key+"."+val for key,vals in six.iteritems(config_schema_dict) for val in vals]

# define schema of config parameters
config_schema_dict = {
    "dataset":  ["path","signal","background","sample_fractions"],
    "features": ["uniform","weight","mT","train","spectator"],
    "training": ["size","signal_id_method","signal_weight_method","weights","algorithms"],
    "hyper":    ["max_depth","n_estimators","subsample","learning_rate","min_samples_leaf","fl_coefficient","power","uniform_label","n_bins","uloss","batchSize","num_of_layers_features","num_of_layers_tag","num_of_layers_pT","num_of_nodes","dropout","epochs","lambdaTag","lambdaReg","lambdaGR","pTBins","n_pTBins"],
}
config_schema = make_schema(config_schema_dict)

# include a default value and some required
config_defaults = {
    "dataset.path": None, "dataset.background": None, "dataset.signal": None
}
