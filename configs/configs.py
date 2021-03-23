import six

def make_schema(config_schema_dict):
    return [key+"."+val for key,vals in six.iteritems(config_schema_dict) for val in vals]

# define schema of config parameters
config_schema_dict = {
    "dataset":  ["path","signal","background"],
    "features": ["uniform","train","spectator"],
    "training": ["size","signal_id_method","signal_weight_method","weights","algorithms"],
    "hyper":    ["max_depth","n_estimators","subsample","learning_rate","min_samples_leaf","fl_coefficient","power","uniform_label","n_bins","uloss"],
}
config_schema = make_schema(config_schema_dict)

# include a default value and some required
config_defaults = {
    "dataset.path": None, "dataset.background": None, "dataset.signal": None
}

