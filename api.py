# NOTE! For research purposes, every investor info is predated by 3 years.
# If you want to get the most accurate results of the model, you need to provide accurate
# and up-to-date investment data in dataset/originals.csv file, and set up the constants
# such that 
# Then you can call run_analysis.py to build the models with the new data 
# load_strat(strat_name) loads a complete strategy (with data type to use, model to use, and combination method to use)

import os
import pickle
import pandas as pd
from constants import * 
from combination_methods import *
from test_models import add_predictions_to_df

cutoff_points   = []
model           = None
input_df        = None
method          = None
input_data_type = ''

def load_strat(_strat_name):
    global cutoff_points, model, input_data_type, input_df, method
    strat_name = _strat_name
    model_name, method_name = get_model_and_method_names(strat_name)

    model_data_name, model_type_name = get_model_data_and_model_type_names(model_name)

    cutoff_points = load_cutoff_points(strat_name)
    method = load_method(method_name)
    model = load_model(model_name)
    input_df = load_input_df(model_data_name)
    input_data_type = 'coinvestors' if model_data_name == 'coinvestors' else 'investors'
    index = 'pair' if model_data_name == 'coinvestors' else 'investor_uuid'
    input_df.set_index(index, inplace=True)

    input_df = add_predictions_to_df(input_df, model)



# Takes a list of investor_uuids of an investment, and returns the 
# suggested decision (True buy, False sell) and probabilty
def get_prediction(investor_uuids):
    def get_pairs(investor_uuids):
        for i, investor_a in enumerate(investor_uuids):
            for investor_b in investor_uuids[i+1:]:
                if investor_a < investor_b:
                    investor_a, investor_b = investor_b, investor_a
                yield f"{investor_a},{investor_b}"
    
    if input_data_type == 'coinvestors':
        return get_prediction_with_selection(investor_uuids, get_pairs)
    else:
        return get_prediction_with_selection(investor_uuids, lambda x: x)


def get_prediction_with_selection(data, get_idx):
    idx_predictions = []
    for idx in get_idx(data):
        if idx in input_df.index:
            idx_predictions.append(input_df.at[idx, 'prediction'])
    if not idx_predictions:
        return False, 0.97

    prediction = method(idx_predictions)
    offset = abs(min(prediction, cutoff_points[1])) + 0.1
    prediction += offset
    cp = [cutoff_points[0] + offset, cutoff_points[1] + offset]
    if prediction > cp[0]:
        return True, float(min(1, 0.5 * prediction / cp[0]))
    if prediction > cutoff_points[1]:
        return False, float(0.7 * (1 - (prediction - cp[1]) / (cp[0] - cp[1])))
    return False, float(min(1, 0.7 * (cp[1] -  prediction / cp[1])))


def load_model(model_name):
    model_filename = f"models/{model_name}.pkl"
    assert(os.path.exists(model_filename))
    with open(model_filename, "rb") as f:
        return pickle.load(f)
    

def load_cutoff_points(strat_name):
    cutoff_points_df = pd.read_csv(CUTOFF_POINTS_FILENAME)
    cutoff_points_df.set_index('strat_name', inplace=True)
    cutoff_point_pairs = []
    for key, value in cutoff_points_df.loc[strat_name].to_dict().items():
        if not key.isnumeric():
            continue
        cutoff_point_pairs.append((int(key), value))
    cutoff_point_pairs.sort()

    cutoff_points = []
    for cutoff_point in cutoff_point_pairs:
        cutoff_points.append(cutoff_point[1])
    return cutoff_points


def get_model_and_method_names(strat_name):
    method_name = strat_name.split('-')[-1]
    model_name = '-'.join(strat_name.split('-')[:-1])
    return model_name, method_name


def get_model_data_and_model_type_names(model_name):
    model_type_name = model_name.split('-')[-1]
    model_data_name = '-'.join(model_name.split('-')[:-1])
    return model_data_name, model_type_name

def load_input_df(model_data_name):
    if model_data_name == 'investors':
        return pd.read_csv(INPUT_INVESTORS_FILENAME)
    if model_data_name == 'investors-with-graph-data':
        return pd.read_csv(INPUT_INVESTORS_WITH_GRAPH_DATA_FILENAME)
    if model_data_name == 'coinvestors':
        return pd.read_csv(INPUT_INVESTOR_PAIRS_FILENAME)


def load_method(method_name):
    if method_name == 'best':
        return best
    if method_name == 'mean':
        return mean
    if method_name == 'median':
        return median
    if method_name == 'mean_top':
        return mean_top
    if method_name == 'median_top':
        return median_top


if __name__ == "__main__":
    load_strat('investors-with-graph-data-linear-mean')
    investor_uuids = [
       "f6f889a7-af59-2363-8984-addadc4da1b2",
       "393b78ac-942d-43f5-af46-c768fdcfc4c6",
       "87b16fdf-a1c8-4ec0-ae9c-323b64cfa70d",
       "40e3b902-7127-3d71-b67b-ec4a2207afe3",
       "73633ee4-ea65-2967-6c5d-9b5fec7d2d5e"
    ]
    print(get_prediction(investor_uuids))
