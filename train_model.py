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


def main():
    pd_set_show_full_content()
    train_test_cycle(
        INPUT_INVESTORS_FILENAME, 
        OUTPUT_INVESTORS_TRAIN_FILENAME, 
        lambda x: LinearRegression(), 
        'investor_uuid', 
        'models/investors-linear.pkl'
    )

    train_test_cycle(
        INPUT_INVESTOR_PAIRS_FILENAME,
        OUTPUT_INVESTOR_PAIRS_TRAIN_FILENAME,
        lambda x: LinearRegression(),
        'pair',
        'models/coinvestors-linear.pkl'
    )

    train_test_cycle(
        INPUT_INVESTORS_FILENAME, 
        OUTPUT_INVESTORS_TRAIN_FILENAME, 
        lambda x: MLPRegressor(hidden_layer_sizes=(x * x, x * x, x)), 
        'investor_uuid', 
        'models/investors-neural.pkl'
    )

    train_test_cycle(
        INPUT_INVESTOR_PAIRS_FILENAME,
        OUTPUT_INVESTOR_PAIRS_TRAIN_FILENAME,
        lambda x: MLPRegressor(hidden_layer_sizes=(x * x, x * x, x)),
        'pair',
        'models/coinvestors-neural.pkl'
    )


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