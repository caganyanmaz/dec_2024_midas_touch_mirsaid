
import pandas as pd
import numpy as np
from constants import *
from utils import *


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
            

def main():
    pd_set_show_full_content()
    investment_df = pd.read_csv(INPUT_INVESTMENTS_FILENAME)
    investment_out_df = pd.read_csv(OUTPUT_INVESTMENTS_FILENAME)
    coinvestor_df = create_investor_pair_df(investment_df)
    coinvestor_out_df = create_investor_pair_df(investment_out_df)
    merged_df = coinvestor_df.merge(coinvestor_out_df, on='pair', how='inner', suffixes=('_train', '_test'))
    print(merged_df.corr(numeric_only=True))

    best_picks = pick_best(investment_out_df, coinvestor_df)
    high_success = 0
    any_success  = 0
    for (_, pick) in best_picks:
        print(pick.total_funding_usd)
        if is_250m(pick):
            high_success += 1
        if is_25m(pick):
            any_success += 1
    print(f"Picks: 250m in first 3 years {high_success}%, 25m in first 3 years: {any_success}%)")
    high_success_total = 0
    any_success_total  = 0
    for investment in investment_out_df.itertuples():
        if is_250m(investment):
            high_success_total += 1
        if is_25m(investment):
            any_success_total += 1
    high_sucess_rate = 100 * high_success_total / len(investment_out_df.index)
    any_success_rate = 100 * any_success_total / len(investment_out_df.index)
    print(f"Total 250m in first 3 years {high_sucess_rate}%, 25m in first 3 years: {any_success_rate}%")



def create_investor_pair_df(investment_df):
    coinvestment_pairs = get_coinvestment_pairs(investment_df)
    coinvestor_pair_df = pd.DataFrame({'pair': coinvestment_pairs}, index=coinvestment_pairs)
    extract_features(coinvestor_pair_df, investment_df)
    coinvestor_pair_df['25m_rate' ] = coinvestor_pair_df['25m_count' ] / coinvestor_pair_df['investment_count']
    coinvestor_pair_df['100m_rate'] = coinvestor_pair_df['100m_count'] / coinvestor_pair_df['investment_count']
    coinvestor_pair_df['250m_rate'] = coinvestor_pair_df['250m_count'] / coinvestor_pair_df['investment_count']
    coinvestor_pair_df['success_rate'] = coinvestor_pair_df['25m_rate'] * WEIGHTS['25m'] + coinvestor_pair_df['100m_rate'] * WEIGHTS['100m'] + coinvestor_pair_df['250m_rate'] * WEIGHTS['250m']
    return coinvestor_pair_df

# Things to get from investments:
# Success?
def extract_features(coinvestor_pair_df, investment_df):
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, 'investment_count', lambda x: True)
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, '25m_count', is_25m)
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, '100m_count', is_100m)
    add_coinvestment_counts_with_condition(coinvestor_pair_df, investment_df, '250m_count', is_250m)


def get_coinvestment_pairs(investment_df):
    coinvestment_pairs = set()
    for investment in investment_df.itertuples():
        for investor_a in get_investors(investment):
            for investor_b in get_investors(investment):
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