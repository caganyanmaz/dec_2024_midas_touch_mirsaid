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
    input_investors_df = pd.read_csv(INPUT_INVESTORS_FILENAME)
    output_investors_train_df = pd.read_csv(OUTPUT_INVESTORS_TRAIN_FILENAME)
    model = train(input_investors_df, output_investors_train_df, 'investor_uuid')
    res = test(input_investors_df, output_investors_train_df, model, 'investor_uuid')
    print(res)
    with open('models/investors-linear.pkl','wb') as f:
        pickle.dump(model, f)

    input_coinvestors_df = pd.read_csv(INPUT_INVESTOR_PAIRS_FILENAME)
    output_coinvestors_train_df = pd.read_csv(OUTPUT_INVESTOR_PAIRS_TRAIN_FILENAME)
    model = train(input_coinvestors_df, output_coinvestors_train_df, 'pair')

def train(input_df, output_df, index):        
    training_data = create_training_data(input_df, output_df, index)
    no_input = training_data.shape[1] - 1
    X = training_data.iloc[:, 0:no_input]
    Y = training_data.iloc[:, no_input]
    #model = MLPRegressor(hidden_layer_sizes=(no_input * no_input, no_input * no_input, no_input), max_iter=1000)
    model = LinearRegression()
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