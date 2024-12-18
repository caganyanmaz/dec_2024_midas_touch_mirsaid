# --------- FILE TO RUN AFTER THE DATASET_PREP.PY FILE TO CALCULATE FEATURES ----------------
# This code file has the code to calculate and extract features from the prepped dataset
# It is vital that all the files are run in order. dataset_prep.py -> feature_eng.py

# NOTE: ORDER OF FEATURE CALCULATION CAN BE RESHUFFLED LOGICALLY TO IMPROVE EFFICIENCY - FUTURE IMPROVEMENT

import pandas as pd
import random
from constants import *


# Features to add:
# 1. Annualised investment count
# 2. 250M+ rate since 2013, 100M+ rate since 2017, 25M rate since 2019
# 3. Investing experience (till 2021)
# 4. Number of categories (broad and specific) invested into by each investor
# 5. Weighted outlier score
# 6. Combining buckets
# 7. grad_rate? (look what this is)
# 8. Remove small investors

def main():
    global WEIGHTS
    best_weights = {"250m": 0.34, "100m": 0.33, "25m": 0.33}
    WEIGHTS = best_weights
    best_val = test()
    for i in range(1000):
        print(WEIGHTS, best_weights, best_val)
        WEIGHTS = create_random_weights()
        val = test()
        if val > best_val:
            best_weights = WEIGHTS

def test():
    pd_set_show_full_content()
    training_investments_df = pd.read_csv(TRAINING_INVESTMENTS_FILENAME)
    training_investors_df   = pd.read_csv(INVESTORS_FILENAME)
    testing_investors_df    = training_investors_df.copy()
    training_investors_df = extract_data_from_investments(training_investors_df, training_investments_df)
    create_features(training_investors_df)

    testing_investments_df  = pd.read_csv(TESTING_INVESTMENTS_FILENAME)
    testing_investors_df = extract_data_from_investments(testing_investors_df, testing_investments_df)
    create_features(testing_investors_df)


    merged = pd.DataFrame.merge(training_investors_df, testing_investors_df,
                        on=['investor_uuid'],
                        how='inner',
                       suffixes=('_train', '_test'))
    corr = merged.corr(numeric_only=True) 
    return corr.at['weighted_outlier_score_test', 'weighted_outlier_score_train']


def create_random_weights():
    a = random.random()
    b = random.random()
    c = random.random()
    s = a + b + c
    return {"250m": a / s, "100m": b / s, "25m": c / s}



#Also drops investors with zero investment
def extract_data_from_investments(investors_df, investments_df):
    investor_columns = get_investor_columns(investors_df)

    add_investment_counts_with_condition(investors_df, investments_df, investor_columns, 'investment_count', lambda x: True)
    def is_25m(investment):
        return investment.total_funding_usd >= 25000000 and investment.founded_year >= YEAR_THRESHOLDS['25m']
    def is_100m(investment):
        return investment.total_funding_usd >= 100000000 and investment.founded_year >= YEAR_THRESHOLDS['100m']
    def is_250m(investment):
        return investment.total_funding_usd >= 250000000 and investment.founded_year >= YEAR_THRESHOLDS['250m']
    add_investment_counts_with_condition(investors_df, investments_df, investor_columns, '25m_count', is_25m)
    add_investment_counts_with_condition(investors_df, investments_df, investor_columns, '100m_count', is_100m)
    add_investment_counts_with_condition(investors_df, investments_df, investor_columns, '250m_count', is_250m)
    add_earliest_investment_experiences(investors_df, investments_df, investor_columns)

    add_earliest_investment_experiences(investors_df, investments_df, investor_columns)
    add_category_counts(investors_df, investments_df, investor_columns, 'category_groups_list', 'broad_category_count')
    add_category_counts(investors_df, investments_df, investor_columns, 'category_list', 'specific_category_count')
    investors_df = investors_df[investors_df['investment_count'] > 0]
    return investors_df



def create_features(investors_df):
    calculate_annualised_investment_counts(investors_df)
    calculate_success_rates(investors_df)
    calculate_investment_experiences(investors_df)
    calculate_investment_diversities(investors_df)
    calculate_weighted_outlier_scores(investors_df)


def calculate_annualised_investment_counts(investors_df):
    investors_df['annualised_investment_count'] = investors_df['investment_count'].apply(annualise)


def calculate_success_rates(investors_df):
    investors_df['25m_rate' ] = investors_df['25m_count' ] / investors_df['investment_count']
    investors_df['100m_rate'] = investors_df['100m_count'] / investors_df['investment_count']
    investors_df['250m_rate'] = investors_df['250m_count'] / investors_df['investment_count']
    
def calculate_investment_experiences(investors_df):
    investors_df['experience'] = (2024 - investors_df['earliest_investment_experience']) / (2024 - TRAINING_START_YEAR)


def calculate_investment_diversities(investors_df):
    investors_df['broad_diversity']    = (investors_df['broad_category_count'] ** 2) / investors_df['investment_count']
    normalize_column(investors_df, 'broad_diversity')
    investors_df['specific_diversity'] = (investors_df['specific_category_count'] ** 2) / investors_df['investment_count']
    normalize_column(investors_df, 'specific_diversity')


def calculate_weighted_outlier_scores(investors_df):
    investors_df['weighted_outlier_score'] = investors_df['250m_rate'] * WEIGHTS['250m'] + investors_df['100m_rate'] * WEIGHTS['100m'] + investors_df['25m_rate'] * WEIGHTS['25m']


def annualise(num_investments):
    # Annualising according to training years set
    return num_investments / (TRAINING_END_YEAR - TRAINING_START_YEAR + 1)


def pd_set_show_full_content():
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None) # Show full content of each column


def add_investment_counts_with_condition(investors_df, investments_df, investor_columns, column_name, condition):
    investment_counts = [0] * len(investors_df.index)
    for investment in investments_df.itertuples():
        if not condition(investment):
            continue

        for investor in get_investors(investment, investor_columns):
            investment_counts[investor] += 1

    investors_df[column_name] = investment_counts


def add_earliest_investment_experiences(investors_df, investments_df, investor_columns):
    earliest_investment_experiences = [TRAINING_TEST_SPLIT_YEAR] * len(investors_df.index)
    for investment in investments_df.itertuples():
        for investor in get_investors(investment, investor_columns):
            earliest_investment_experiences[investor] = min(earliest_investment_experiences[investor], investment.founded_year)
    investors_df['earliest_investment_experience'] = earliest_investment_experiences


def add_category_counts(investors_df, investments_df, investor_columns, category_column_name, new_column_name):
    investor_categories = [set() for i in investors_df.index]
    for investment in investments_df.itertuples():
        investment_categories = getattr(investment, category_column_name)
        for investor in get_investors(investment, investor_columns):
            for investment_category in investment_categories.split(','):
                investor_categories[investor].add(investment_category.strip())
    investor_category_counts = list(map(len, investor_categories))
    investors_df[new_column_name] = investor_category_counts


def get_investors(investment, investor_columns):
    for investor_raw in investment.investor_uuids.split(','):
        investor = investor_raw.strip()
        if investor in investor_columns:
            yield investor_columns[investor]


def get_investor_columns(investors_df):
    investor_columns = {}
    for i in investors_df.index:
        investor_uuid = investors_df.at[i, 'investor_uuid']
        investor_columns[investor_uuid] = i
    return investor_columns


def normalize_column(df, column_name):
    df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())


        

if __name__ == "__main__":
    main()
