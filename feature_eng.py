# --------- FILE TO RUN AFTER THE DATASET_PREP.PY FILE TO CALCULATE FEATURES ----------------
# This code file has the code to calculate and extract features from the prepped dataset
# It is vital that all the files are run in order. dataset_prep.py -> feature_eng.py

# NOTE: ORDER OF FEATURE CALCULATION CAN BE RESHUFFLED LOGICALLY TO IMPROVE EFFICIENCY - FUTURE IMPROVEMENT

import pandas as pd
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
    pd_set_show_full_content()
    training_investments_df = pd.read_csv(TRAINING_INVESTMENTS_FILENAME)
    investors_df            = pd.read_csv(INVESTORS_FILENAME)
    print('Loaded files')
    print(training_investments_df)
    investor_columns = get_investor_columns(investors_df)
    print('Got investor columns')

    add_investment_counts_with_condition(investors_df, training_investments_df, investor_columns, 'investment_count', lambda x: True)
    def is_25m(investment):
        return investment.total_funding_usd >= 25000000 and investment.founded_year >= YEAR_THRESHOLDS['25m']
    def is_100m(investment):
        return investment.total_funding_usd >= 100000000 and investment.founded_year >= YEAR_THRESHOLDS['100m']
    def is_250m(investment):
        return investment.total_funding_usd >= 250000000 and investment.founded_year >= YEAR_THRESHOLDS['250m']
    add_investment_counts_with_condition(investors_df, training_investments_df, investor_columns, '25m_count', is_25m)
    add_investment_counts_with_condition(investors_df, training_investments_df, investor_columns, '100m_count', is_100m)
    add_investment_counts_with_condition(investors_df, training_investments_df, investor_columns, '250m_count', is_250m)
    add_earliest_investment_experiences(investors_df, training_investments_df, investor_columns)

    print(investors_df)
    print(investors_df['250m_count'].sum())
    add_earliest_investment_experiences(investors_df, training_investments_df, investor_columns)
    investors_df['earliest_investment_experience'] = 2024
    investor_broad_categories    = {investor_name: set() for investor_name in investors_df['investor_name']} 
    investor_specific_categories = investor_broad_categories.copy()
    

    for investment in training_investments_df:
        print(investment)
        break
    
    


    """
    # All zero scores get L0
    zero_df['outlier_bucket'] = 'L0'

    # Divide the nonzero scores into 5 buckets: L1–L5
    nonzero_df['outlier_bucket'] = pd.qcut(
        nonzero_df['outlier_score'],
        q=5,
        labels=['L1', 'L2', 'L3', 'L4', 'L5']
    )

    # Combine back into one DataFrame
    all_investors_df = pd.concat([zero_df, nonzero_df], ignore_index=True)
    all_investors_df = all_investors_df.sort_values('outlier_score', ascending=True)

    # Group by outlier_bucket and get min/max outlier_score
    quintile_ranges = all_investors_df.groupby('outlier_bucket')['outlier_score'].agg(['min', 'max']).reset_index()
    all_investors_df.to_csv(ALL_INVESTORS_FILENAME, index=False)



    # FEATURE 6: combining both bucket types to see how many investors in each bucket and their combined outlier score

    # Filter out L0 investors
    filtered_df = all_investors_df[all_investors_df['outlier_bucket'] != 'L0']

    # Group by both bucket types and compute the required statistics
    bucketstats_df = filtered_df.groupby(['outlier_bucket', 'focus_classification']).agg(
        count=('outlier_score', 'size'),
        mean_outlier_score=('outlier_score', 'mean'),
        median_outlier_score=('outlier_score', 'median'),
        stddev_outlier_score=('outlier_score', 'std')
    ).reset_index()

    bucketstats_df.to_csv(BUCKETS_FILENAME, index=False)


    # FEATURE 7: add average recent_grad_rate to each of the 25 bucket classifications

    # Calculate avg_recent_grad_rate for each combination of outlier_bucket and focus_classification
    avg_grad_rate = all_investors_df.groupby(['outlier_bucket', 'focus_classification'])['percent_25m_success'].mean().reset_index(name='avg_recent_grad_rate')

    # Merge avg_recent_grad_rate into bucketstats_df
    bucketstats_df = bucketstats_df.merge(
        avg_grad_rate,
        on=['outlier_bucket', 'focus_classification'],
        how='left'
    )

    # Save the updated bucketstats_df to a file or preview the result
    bucketstats_df.to_csv(BUCKETS_FILENAME, index=False)
    print(bucketstats_df)


    # FEATURE 8: remove all 1-investment investors from the all_investors_df as they don't have any coinvestments

    # Filter out all non-L0 rows where num_investments > 1
    kg_investors_df = all_investors_df[(all_investors_df['num_investments_2013'] > 1) &
                                       (all_investors_df['outlier_bucket'] != "L0")]

    # Save the filtered and modified DataFrame
    kg_investors_df.to_csv(KG_INVESTORS_FILENAME, index=False)
    """


def calculate_success_percentages(investor_uuid, total_investments, coinvestor_df):
    # Filter coinvestor_df where the investor_uuid is present in the investor_uuids column
    relevant_startups = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    # Count the number of successes
    count_250m = relevant_startups['ultimate_outlier_success_250_mil_raise'].sum()
    count_100m = relevant_startups['interim_success_100_mil_founded_year_2019_or_above'].sum()
    count_25m = relevant_startups['recent_success_25_mil_raise_founded_year_2022_or_above'].sum()
    # Calculate percentages (handle divide by zero)
    if total_investments > 0:
        return (count_250m / total_investments * 100,
                count_100m / total_investments * 100,
                count_25m / total_investments * 100)
    else:
        return (0, 0, 0)


def calculate_balanced_outlier_score(row):    
    # Investment factor (how much to penalise investors with little investments)
    if row['num_investments_2013'] <= 5:
        investment_factor = (row['num_investments_2013'] - 1) / 4
    else:
        investment_factor = 1
    
    # Experience factor (how much to penalise investors that are very new to the market)
    if row['investing_experience'] <= 3:
        experience_factor = row['investing_experience'] / 3
    else:
        experience_factor = 1
    
    # Adjusted success probabilities (only scales down high probabilities, doesn't increase any previous probability calculations)
    adjusted_250m = row['percent_250m_success'] * investment_factor * experience_factor
    adjusted_100m = row['percent_100m_success'] * investment_factor * experience_factor
    adjusted_25m = row['percent_25m_success'] * investment_factor * experience_factor
    
    # Outlier score final
    outlier_score = WEIGHTS['weight_250m'] * adjusted_250m + WEIGHTS['weight_100m'] * adjusted_100m + WEIGHTS['weight_25m'] * adjusted_25m
    return outlier_score


def num_investments(investor_uuid, coinvestor_df):
    # Filter rows containing this investor
    relevant = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    return len(relevant)


def annualise(num_investments):
    # Annualising according to training years set
    return num_investments / (TRAINING_END_YEAR - TRAINING_START_YEAR + 1)


def get_first_investment_year(investor_uuid, coinvestor_df):
    # Filter coinvestor_df for rows containing the investor_uuid
    relevant_rows = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    if not relevant_rows.empty:
        # Assuming 'founded_year' is the year column in coinvestor_df
        return relevant_rows['founded_year'].min()  # Get the earliest year
    else:
        return 2024  # No investments found, assume 0 experience (shouldnt be many investors like this)


def get_investor_categories_broad(investor_uuid, coinvestor_df):
    # Filter coinvestor_df where the investor_uuid is present in the investor_uuids column
    relevant_startups = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    # Extract the unique categories from the "category_groups_list" column
    categories = relevant_startups['category_groups_list'].str.split(',').explode().unique()
    # Convert all categories to strings and filter out None/NaN values
    categories = [str(category) for category in categories if pd.notna(category)]
    # Return the list of categories as a CSV string and the count of unique categories
    return ', '.join(categories), len(categories)


def get_investor_categories_specific(investor_uuid, coinvestor_df):
    # Filter coinvestor_df where the investor_uuid is present in the investor_uuids column
    relevant_startups = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    # Extract the unique categories from the "category_list" column
    categories = relevant_startups['category_list'].str.split(',').explode().unique()
    # Convert all categories to strings and filter out None/NaN values
    categories = [str(category) for category in categories if pd.notna(category)]
    # Return the list of categories as a CSV string and the count of unique categories
    return ', '.join(categories), len(categories)


def pd_set_show_full_content():
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None) # Show full content of each column


def add_investment_counts_with_condition(investors_df, training_investments_df, investor_columns, column_name, condition):
    investment_counts = [0] * len(investors_df.index)
    for investment in training_investments_df.itertuples():
        if not condition(investment):
            continue

        for investor in get_investors(investment, investor_columns):
            investment_counts[investor] += 1

    investors_df[column_name] = investment_counts


def add_earliest_investment_experiences(investors_df, training_investments_df, investor_columns):
    earliest_investment_experiences = [TRAINING_TEST_SPLIT_YEAR] * len(investors_df.index)
    for investment in training_investments_df.itertuples():
        for investor in get_investors(investment, investor_columns):
            earliest_investment_experiences[investor] = min(earliest_investment_experiences, investment.founded_year)
    investors_df['earliest_investment_experiences'] = earliest_investment_experiences


def get_investors(investment, investor_columns):
    for investor_raw in investment.investor_uuids.split(','):
        investor = investor_raw.strip()
        if investor in investor_columns:
            yield investor_columns[investor]


def get_investor_columns(investors_df):
    print(len(investors_df.index))
    investor_columns = {}
    print(investors_df)
    for i in investors_df.index:
        investor_uuid = investors_df.at[i, 'investor_uuid']
        investor_columns[investor_uuid] = i
    return investor_columns


        

if __name__ == "__main__":
    main()
