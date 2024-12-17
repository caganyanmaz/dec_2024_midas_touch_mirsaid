import pandas as pd

# TODO: Complete this later

def calculate_mean_successes(training_investment_df, test_investment_df):

    MEAN_PROB_SUCCESS = longterm_df['success_rate'].mean() # real-world chance of randomly investing in a startup that raises $250M+ over its lifetime, 2013-now
    MEAN_PROB_EARLY_SUCCESS = shortterm_df['success_rate'].mean() # chance of randomly investing in a 2022-now founded startup that raises $25M+
    REAL_WORLD_PROB = 1.9 # 1.9 percent chance of randomly picking a 'successful' startup, where successful is defined as an outlier ($250M+ raised)





