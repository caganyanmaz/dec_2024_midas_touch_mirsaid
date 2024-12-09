# --------- FILE TO RUN AFTER THE DATASET_PREP.PY FILE TO CALCULATE FEATURES ----------------
# This code file has the code to calculate and extract features from the prepped dataset
# It is vital that all the files are run in order. dataset_prep.py -> feature_eng.py

from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import openai

load_dotenv() #make a .env file and store OPENAI_APIKEY in there to retrieve it here
openai_apikey = os.getenv("OPENAI_APIKEY")

REAL_WORLD_PROB = 1.9 # 1.9 percent chance of randomly picking a 'successful' startup, where successful is defined as an outlier ($250M+ raised)

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None) # Show full content of each column


# filepaths for the new cleaned datasets to work on
filepath1 = "dataset/coinvestor_clean.csv"
filepath2 = "dataset/long_term_clean.csv"
filepath3 = "dataset/short_term_clean.csv"
filepath4 = "dataset/investors.csv"
filepath5 = "dataset/startups.csv"

coinvestor_df = pd.read_csv(filepath1)
longterm_df = pd.read_csv(filepath2)
shortterm_df = pd.read_csv(filepath3)

investors_df = pd.read_csv(filepath4)
startups_df = pd.read_csv(filepath5)


MEAN_PROB_SUCCESS = longterm_df['success_rate'].mean() # real-world chance of randomly investing in a startup that raises $250M+ over its lifetime, 2013-now
MEAN_PROB_EARLY_SUCCESS = shortterm_df['success_rate'].mean() # chance of randomly investing in a 2022-now founded startup that raises $25M+

'''
#FEATURE 1: annualised investments (how many investments a given investor makes per year)

def num_investments(investor_uuid):
    # Filter rows containing this investor
    relevant = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    # Deduplicate by org_uuid to ensure each startup is counted once
    relevant = relevant.drop_duplicates(subset='org_uuid')
    return len(relevant)

def annualise(num_investments):
    # 2013 to 2024 is a 12-year span (assuming inclusive counting 2013 through 2024)
    return num_investments / 12

# mapping the num_investments function to the coinvestments table to find how many of the startups each of the investors had invested in (since 2013, as that is the oldest data)
# annualising the investments made per year since the dataset initial year, 2013 to 2024

investors_df['num_investments_2013'] = investors_df['investor_uuid'].apply(num_investments)
investors_df['annualised_investments_2013'] = investors_df['num_investments_2013'].apply(annualise)
investors_df.to_csv("dataset/investors.csv")


# FEATURE 2: 250M+ outlier rate since 2013, 100M+ outlier rate since 2019, 25M+ outlier rate since 2022
def calculate_success_percentages(investor_uuid, total_investments):
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


# Apply the function to each investor_uuid and unpack the results into new columns
investors_df[['percent_250m_success', 'percent_100m_success', 'percent_25m_success']] = investors_df.apply(
    lambda row: pd.Series(calculate_success_percentages(row['investor_uuid'], row['num_investments_2013'])),
    axis=1
)
investors_df.to_csv("dataset/investors.csv", index=False)

#FEATURE 3: number of categories (broad and specific) invested into by each investor

# Define a function to extract broad categories and count them
def get_investor_categories_broad(investor_uuid):
    # Filter coinvestor_df where the investor_uuid is present in the investor_uuids column
    relevant_startups = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    # Extract the unique categories from the "category_groups_list" column
    categories = relevant_startups['category_groups_list'].str.split(',').explode().unique()
    # Convert all categories to strings and filter out None/NaN values
    categories = [str(category) for category in categories if pd.notna(category)]
    # Return the list of categories as a CSV string and the count of unique categories
    return ', '.join(categories), len(categories)

# Define a function to extract specific categories and count them
def get_investor_categories_specific(investor_uuid):
    # Filter coinvestor_df where the investor_uuid is present in the investor_uuids column
    relevant_startups = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    # Extract the unique categories from the "category_list" column
    categories = relevant_startups['category_list'].str.split(',').explode().unique()
    # Convert all categories to strings and filter out None/NaN values
    categories = [str(category) for category in categories if pd.notna(category)]
    # Return the list of categories as a CSV string and the count of unique categories
    return ', '.join(categories), len(categories)

# Apply the functions to add the broad and specific categories to the investors_df
investors_df[['categories_broad', 'categories_broad_count']] = investors_df['investor_uuid'].apply(
    lambda uuid: pd.Series(get_investor_categories_broad(uuid))
)

investors_df[['categories_specific', 'categories_specific_count']] = investors_df['investor_uuid'].apply(
    lambda uuid: pd.Series(get_investor_categories_specific(uuid))
)
investors_df.to_csv("dataset/investors.csv", index=False)


# Bucketing investors based on the diversity of their sector/industry investments count
investors_df = investors_df.sort_values(by=['categories_broad_count', 'categories_specific_count'], ascending=[True, True])
# Divide the DataFrame into quintiles
investors_df['focus_classification'] = pd.qcut(
    investors_df['categories_broad_count'], 
    q=5,  # Number of buckets
    labels=['Specialist', 'Focused', 'Balanced', 'Diverse', 'Universalist']  # Labels for focus_classifications
)
# Extract min and max for each focus_classification
quintile_ranges = investors_df.groupby('focus_classification')['categories_broad_count'].agg(['min', 'max']).reset_index()
print(quintile_ranges)
# focus_classification  min  max
# 0        Specialist    5   16
# 1           Focused   17   21
# 2          Balanced   22   26
# 3           Diverse   27   34
# 4      Universalist   35   49
investors_df.to_csv("dataset/investors.csv", index=False)
'''

# grouped_rates = investors_df.groupby('focus_classification')[['percent_250m_success', 'percent_100m_success', 'percent_25m_success']].mean()
# print(grouped_rates)
# #                       percent_250m_success  percent_100m_success   percent_25m_success
# # focus_classification
# # Specialist                        3.632572              4.974476              7.548869
# # Focused                           5.825818              7.420882              8.060355
# # Balanced                         10.713812             11.219362              8.025647
# # Diverse                          20.005525              9.869131              6.818255
# # Universalist                     32.275928             13.394807              7.032856     


# FEATURE 4: calculate a custom overall outlier score to bucket investors by outlier rate, taking into account 250m, 100m and 25m success, but weighted (0.85, 0.10, 0.05)
# Calculate the weighted outlier score for each investor
investors_df['outlier_score'] = (
    0.85 * investors_df['percent_250m_success'] +
    0.1 * investors_df['percent_100m_success'] +
    0.05 * investors_df['percent_25m_success']
)

# Sort the DataFrame by outlier_score in ascending order
investors_df = investors_df.sort_values('outlier_score', ascending=True)

# Split into zero and nonzero outlier scores
zero_df = investors_df[investors_df['outlier_score'] == 0].copy()
nonzero_df = investors_df[investors_df['outlier_score'] > 0].copy()

# All zero scores get L0
zero_df['outlier_bucket'] = 'L0'

# Divide the nonzero scores into 5 buckets: L1–L5
nonzero_df['outlier_bucket'] = pd.qcut(
    nonzero_df['outlier_score'],
    q=5,
    labels=['L1', 'L2', 'L3', 'L4', 'L5']
)

# Combine back into one DataFrame
investors_df = pd.concat([zero_df, nonzero_df], ignore_index=True)
investors_df = investors_df.sort_values('outlier_score', ascending=True)

# Group by outlier_bucket and get min/max outlier_score
quintile_ranges = investors_df.groupby('outlier_bucket')['outlier_score'].agg(['min', 'max']).reset_index()
# print(quintile_ranges)
#   outlier_bucket       min        max
# 0             L0  0.000000   0.000000
# 1             L1  0.044248   0.625000
# 2             L2  0.638629   1.818182
# 3             L3  1.826923   3.750000
# 4             L4  3.750000   6.375000
# 5             L5  6.394558  31.666667
investors_df.to_csv(filepath4, index=False)