# --------- FILE TO RUN AFTER THE DATASET_PREP.PY FILE TO CALCULATE FEATURS ----------------
# This code file has the code to calculate and extract features from the prepped dataset
# It is vital that all the files are run in order. dataset_prep.py -> feature_eng.py

# NOTE: ORDER OF FEATURE CALCULATION CAN BE RESHUFFLED LOGICALLY TO IMPROVE EFFICIENCY - FUTURE IMPROVEMENT

import pandas as pd
from utils import *
from constants import *
from sklearn.model_selection import train_test_split


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
    pd_set_show_full_content()
    create_for(INPUT_INVESTMENTS_FILENAME, INPUT_INVESTORS_FILENAME)
    create_for(OUTPUT_INVESTMENTS_FILENAME, OUTPUT_INVESTORS_FILENAME)
    output_investors_df = pd.read_csv(OUTPUT_INVESTORS_FILENAME)
    output_investors_df.set_index('investor_uuid', inplace=True)
    output_investors_train_df, output_investors_test_df = train_test_split(output_investors_df, test_size=TEST_SIZE, random_state=42)
    output_investors_train_df.to_csv(OUTPUT_INVESTORS_TRAIN_FILENAME)
    output_investors_test_df.to_csv(OUTPUT_INVESTORS_TEST_FILENAME)
    

def create_for(investment_filename, investor_filename):
    input_investments_df = pd.read_csv(investment_filename)
    input_investors_df   = pd.read_csv(investor_filename)
    input_investors_df.set_index('investor_uuid', inplace=True)
    input_investors_df = extract_data_from_investments(input_investors_df, input_investments_df)
    create_features(input_investors_df)
    input_investors_df.to_csv(investor_filename)


def get_best_picks(input_investor_df, output_investment_df):
    investment_value = []
    for investment in output_investment_df.itertuples():
        best_value = 0
        for investor in get_investors(investment):
            if investor in input_investor_df.index:
                current_value = input_investor_df.at[investor, 'weighted_success_rate']
                best_value = max(best_value, current_value)
                print('hi')
        investment_value.append((best_value, investment))
    investment_value.sort(reverse=True)
    res = list(map(lambda x: x[1], investment_value[:1000]))
    return res

def get_investor_correlation(input_investor_df, output_investor_df):
    merged = pd.DataFrame.merge(input_investor_df, output_investor_df,
                        on=['investor_uuid'],
                        how='inner',
                       suffixes=('_train', '_test'))
    return merged.corr(numeric_only=True)


def extract_data_from_investments(investors_df, investments_df):
    add_investment_counts_with_condition(investors_df, investments_df, 'investment_count', lambda x: True)

    add_investment_counts_with_condition(investors_df, investments_df, '25m_count', is_25m)
    add_investment_counts_with_condition(investors_df, investments_df, '100m_count', is_100m)
    add_investment_counts_with_condition(investors_df, investments_df, '250m_count', is_250m)
    add_earliest_investment_experiences(investors_df, investments_df)

    add_earliest_investment_experiences(investors_df, investments_df)
    add_category_counts(investors_df, investments_df, 'category_groups_list', 'broad_category_count')
    add_category_counts(investors_df, investments_df, 'category_list', 'specific_category_count')
    #investors_df = investors_df[investors_df['investment_count'] > 0]
    return investors_df


def create_features(investors_df):
    calculate_annualised_investment_counts(investors_df)
    calculate_success_rates(investors_df)
    calculate_investment_experiences(investors_df)
    calculate_investment_diversities(investors_df)
    calculate_weighted_success_rates(investors_df)


def calculate_annualised_investment_counts(investors_df):
    investors_df['annualised_investment_count'] = investors_df['investment_count'].apply(annualise)


def calculate_success_rates(investors_df):
    investors_df['25m_rate' ] = investors_df['25m_count' ] / investors_df['investment_count']
    investors_df['100m_rate'] = investors_df['100m_count'] / investors_df['investment_count']
    investors_df['250m_rate'] = investors_df['250m_count'] / investors_df['investment_count']
    

def calculate_investment_experiences(investors_df):
    investors_df['experience'] = (CURRENT_YEAR - investors_df['earliest_investment_experience']) / (CURRENT_YEAR - INPUT_START_YEAR)


def calculate_investment_diversities(investors_df):
    investors_df['broad_diversity']    = (investors_df['broad_category_count'] ** 2) / investors_df['investment_count']
    normalize_column(investors_df, 'broad_diversity')
    investors_df['specific_diversity'] = (investors_df['specific_category_count'] ** 2) / investors_df['investment_count']
    normalize_column(investors_df, 'specific_diversity')


def calculate_weighted_success_rates(investors_df):
    investors_df['weighted_success_rate'] = investors_df['250m_rate'] * WEIGHTS['250m'] + investors_df['100m_rate'] * WEIGHTS['100m'] + investors_df['25m_rate'] * WEIGHTS['25m']


def annualise(num_investments):
    # Annualising according to training years set
    return num_investments / (INPUT_END_YEAR - INPUT_START_YEAR + 1)
def add_investment_counts_with_condition(investors_df, investments_df, column_name, condition):
    investment_counts = {investor: 0 for investor in investors_df.index}
    for investment in investments_df.itertuples():
        if not condition(investment):
            continue

        for investor in get_investors(investment):
            if investor in investment_counts:
                investment_counts[investor] += 1

    investors_df[column_name] = investment_counts


def add_earliest_investment_experiences(investors_df, investments_df):
    earliest_investment_experiences = {investor: INPUT_OUTPUT_SPLIT_YEAR for investor in investors_df.index}
    for investment in investments_df.itertuples():
        for investor in get_investors(investment):
            if investor in earliest_investment_experiences:
                earliest_investment_experiences[investor] = min(earliest_investment_experiences[investor], investment.founded_year)
    investors_df['earliest_investment_experience'] = earliest_investment_experiences


def add_category_counts(investors_df, investments_df, category_column_name, new_column_name):
    investor_categories = {investor: set() for investor in investors_df.index}
    for investment in investments_df.itertuples():
        investment_categories = getattr(investment, category_column_name)
        for investor in get_investors(investment):
            if investor in investor_categories:
                for investment_category in investment_categories.split(','):
                    investor_categories[investor].add(investment_category.strip())
    investor_category_counts = {}
    for investor, categories in investor_categories.items():
        investor_category_counts[investor] = len(categories)
    investors_df[new_column_name] = investor_category_counts


def normalize_column(df, column_name):
    df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())


if __name__ == "__main__":
    main()
