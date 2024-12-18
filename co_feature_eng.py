# Needs to be run after feature_eng.py
import pandas as pd
import numpy as np
from constants import *
from utils import *
from sklearn.model_selection import train_test_split
            

def main():
    pd_set_show_full_content()
    create_for(INPUT_INVESTMENTS_FILENAME, INPUT_INVESTORS_FILENAME, INPUT_INVESTOR_PAIRS_FILENAME)
    create_for(OUTPUT_INVESTMENTS_FILENAME, OUTPUT_INVESTORS_FILENAME, OUTPUT_INVESTOR_PAIRS_FILENAME)
    output_investor_pairs_df = pd.read_csv(OUTPUT_INVESTOR_PAIRS_FILENAME)
    output_investor_pairs_df.set_index('pair', inplace=True)
    output_investor_pairs_train_df, output_investor_pairs_test_df = train_test_split(output_investor_pairs_df, test_size=TEST_SIZE, random_state=42)
    output_investor_pairs_train_df.to_csv(OUTPUT_INVESTOR_PAIRS_TRAIN_FILENAME)
    output_investor_pairs_test_df.to_csv(OUTPUT_INVESTOR_PAIRS_TEST_FILENAME)


def create_for(investment_filename, investor_filename, investor_pair_filename):
    input_investment_df = pd.read_csv(investment_filename)
    input_investor_df  = pd.read_csv(investor_filename)
    input_investor_df.set_index('investor_uuid', inplace=True)
    investor_pair_df = create_investor_pair_df(input_investment_df, input_investor_df)
    investor_pair_df.to_csv(investor_pair_filename)


# Features I can think of:
# success_rate
# individual_success_rate
def pick_best(investments, coinvestor_df):
    picks = []    
    for investment in investments.itertuples():
        found_investor = False
        best_success_rate = 0
        for investor_a, investor_b in get_investor_pairs(investment):
            pair = f"{investor_a},{investor_b}"
            if pair in coinvestor_df.index:
                found_investor = True
                best_success_rate = max(best_success_rate, coinvestor_df.at[pair, 'success_rate'])
        picks.append((best_success_rate, investment))
    picks.sort(reverse=True)
    return picks[:100]



def create_investor_pair_df(investment_df, investor_df):
    coinvestment_pairs = get_coinvestment_pairs(investment_df, investor_df)
    coinvestor_pair_df = pd.DataFrame({'pair': coinvestment_pairs}, index=coinvestment_pairs)
    extract_features_from_investments(coinvestor_pair_df, investment_df)
    coinvestor_pair_df['25m_rate' ] = coinvestor_pair_df['25m_count' ] / coinvestor_pair_df['investment_count']
    coinvestor_pair_df['100m_rate'] = coinvestor_pair_df['100m_count'] / coinvestor_pair_df['investment_count']
    coinvestor_pair_df['250m_rate'] = coinvestor_pair_df['250m_count'] / coinvestor_pair_df['investment_count']
    coinvestor_pair_df['weighted_success_rate'] = coinvestor_pair_df['25m_rate'] * WEIGHTS['25m'] + coinvestor_pair_df['100m_rate'] * WEIGHTS['100m'] + coinvestor_pair_df['250m_rate'] * WEIGHTS['250m']
    coinvestor_pair_df = coinvestor_pair_df[coinvestor_pair_df['weighted_success_rate'] > 0]
    extract_features_from_investors(coinvestor_pair_df, investor_df)
    return coinvestor_pair_df

# Things to get from investments:
# Success?
def extract_features_from_investments(coinvestor_pair_df, investment_df):
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, 'investment_count', lambda x: True)
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, '25m_count', is_25m)
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, '100m_count', is_100m)
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, '250m_count', is_250m)


def extract_features_from_investors(coinvestor_pair_df, investor_df):
    add_average(coinvestor_pair_df, investor_df, 'experience', 'average_experience') 
    add_average(coinvestor_pair_df, investor_df, 'weighted_success_rate', 'average_success_rate') 
    add_average(coinvestor_pair_df, investor_df, 'specific_diversity', 'average_specific_diversity')
    add_average(coinvestor_pair_df, investor_df, 'broad_diversity', 'average_broad_diversity')


def add_average(coinvestor_pair_df, investor_df, column_name, new_column_name):
    average = {}
    for pair in coinvestor_pair_df.index:
        investor_a, investor_b = pair.split(',')
        average[pair] = (investor_df.at[investor_a, column_name] + investor_df.at[investor_b, column_name]) / 2
    coinvestor_pair_df[new_column_name] = coinvestor_pair_df.index.map(average)



def get_coinvestment_pairs(investment_df, investor_df):
    coinvestment_pairs = set()
    for investment in investment_df.itertuples():
        for investor_a in get_investors(investment):
            if investor_a not in investor_df.index:
                continue
            for investor_b in get_investors(investment):
                if investor_b not in investor_df.index:
                    continue
                if investor_a > investor_b:
                    coinvestment_pairs.add(f"{investor_a},{investor_b}")
    return list(coinvestment_pairs)

def add_coinvestment_counts_with_condition(coinvestor_pair_df, investments_df, column_name, condition):
    coinvestment_counts = {pair: 0 for pair in coinvestor_pair_df.index}
    
    for investment in investments_df.itertuples():
        if not condition(investment):
            continue
        
        for investor_a, investor_b in get_investor_pairs(investment):
            pair = f"{investor_a},{investor_b}" 
            if pair in coinvestment_counts:
                coinvestment_counts[pair] += 1

    coinvestor_pair_df[column_name] = coinvestor_pair_df.index.map(coinvestment_counts)


def get_investor_pairs(investment):
    for investor_a in get_investors(investment):
        for investor_b in get_investors(investment):
            if investor_a > investor_b:
                yield (investor_a, investor_b)


if __name__ == "__main__":
    main()
