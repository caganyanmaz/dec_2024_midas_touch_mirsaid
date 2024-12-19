# The neural network will take investor's (or coinvestor pair's)
# features and try to predict the success probability of the startup
# To pick a startup to invest, we can find every investor (and investor pair)
# get their success probability, and get the max of them (as I think
# a good investor is more influential than a bad one)
import pickle
import pandas as pd
from utils import *
from constants import *
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

RANDOM_SEED = 42

def main():
    pd_set_show_full_content()
    datasets = [
        (INPUT_INVESTORS_FILENAME, OUTPUT_INVESTORS_FILENAME, 'investor_uuid', 'investors'),
        (INPUT_INVESTOR_PAIRS_FILENAME, OUTPUT_INVESTOR_PAIRS_FILENAME, 'pair', 'coinvestors'),
        (INPUT_INVESTORS_WITH_GRAPH_DATA_FILENAME, OUTPUT_INVESTORS_FILENAME, 'investor_uuid', 'investors-with-graph-data')
    ]
    models = [
        (lambda x: LinearRegression(), 'linear'),
        (lambda x: MLPRegressor(hidden_layer_sizes=(x * x, x * x, x), random_state=RANDOM_SEED), 'neural')
    ]
    for (input_file, output_file, index, dataset_name) in datasets:
        for (model_fun, model_name) in models:
            model_name = f'models/{dataset_name}-{model_name}.pkl'
            train_test_cycle(input_file, output_file, model_fun, index, model_name)


def train_test_cycle(input_filename, output_filename, create_model, index, model_name):
    input_df = pd.read_csv(input_filename)
    output_df = pd.read_csv(output_filename)
    model = train(input_df, output_df, index, create_model)
    res = test(input_df, output_df, model, index)
    print(res)
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)

def train(input_df, output_df, index, create_model):        
    training_data = create_training_data(input_df, output_df, index)
    no_input = training_data.shape[1] - 1
    X = training_data.iloc[:, 0:no_input]
    Y = training_data.iloc[:, no_input]
    #model = MLPRegressor(hidden_layer_sizes=(no_input * no_input, no_input * no_input, no_input), max_iter=1000)
    model = create_model(no_input)
    print('Training model...')
    model.fit(X, Y)
    return model

def test(input_df, output_df, model, index):
    test_data = create_training_data(input_df, output_df, index)
    no_input = test_data.shape[1] - 1
    X = test_data.iloc[:, 0:no_input]
    print(X.columns)
    Y = test_data.iloc[:, no_input]
    return model.score(X, Y)


def create_training_data(input_df, output_df, index):
    merged_df = input_df.merge(output_df[[index, 'weighted_success_rate']], on=index, how='inner')
    merged_df.rename(columns={'weighted_success_rate_y': 'result'}, inplace=True)
    merged_df.rename(columns={'weighted_success_rate_x': 'weighted_success_rate'}, inplace=True)
    merged_df.dropna(inplace=True)
    return merged_df.select_dtypes(include=['float64', 'int64'])


if __name__ == "__main__":
    main()