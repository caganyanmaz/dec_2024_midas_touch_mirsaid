# --------- FILE TO RUN AFTER THE DATASET_PREP.PY FILE TO CALCULATE FEATURES ----------------
# This code file has the code to calculate and extract features from the prepped dataset
# It is vital that all the files are run in order. dataset_prep.py -> feature_eng.py

# NOTE: ORDER OF FEATURE CALCULATION CAN BE RESHUFFLED LOGICALLY TO IMPROVE EFFICIENCY - FUTURE IMPROVEMENT

import pandas as pd

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None) # Show full content of each column

# filepaths for the new cleaned datasets to work on
coinvestor_file = "dataset/coinvestor_clean.csv"
longterm_file = "dataset/long_term_clean.csv"
shortterm_file = "dataset/short_term_clean.csv"
all_investors_file = "dataset/all_investors.csv"
startups_file = "dataset/startups.csv"
buckets_file = "dataset/bucket_stats.csv"
kg_investors_file = "dataset/kg_investors.csv"

coinvestor_df = pd.read_csv(coinvestor_file)
longterm_df = pd.read_csv(longterm_file)
shortterm_df = pd.read_csv(shortterm_file)
all_investors_df = pd.read_csv(all_investors_file)
startups_df = pd.read_csv(startups_file)

MEAN_PROB_SUCCESS = longterm_df['success_rate'].mean() # real-world chance of randomly investing in a startup that raises $250M+ over its lifetime, 2013-now
MEAN_PROB_EARLY_SUCCESS = shortterm_df['success_rate'].mean() # chance of randomly investing in a 2022-now founded startup that raises $25M+
REAL_WORLD_PROB = 1.9 # 1.9 percent chance of randomly picking a 'successful' startup, where successful is defined as an outlier ($250M+ raised)

# weights to sum the outlier_rates for each investor (TWEAK THIS TO PLAY WITH THE OUTLIER SCORE BUCKETING)
WEIGHTS = { 
    'weight_250m': 0.85,
    'weight_100m': 0.1,
    'weight_25m': 0.05
    }

#FEATURE 1: adding annualised investments (how many investments a given investor makes per year)

def num_investments(investor_uuid):
    # Filter rows containing this investor
    relevant = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    return len(relevant)

def annualise(num_investments):
    # 2013 to 2024 is a 12-year span (assuming inclusive counting 2013 through 2024)
    return num_investments / 12

# mapping the num_investments function to the coinvestments table to find how many of the startups each of the investors had invested in (since 2013, as that is the oldest data)
# annualising the investments made per year since the dataset initial year, 2013 to 2024

all_investors_df['num_investments_2013'] = all_investors_df['investor_uuid'].apply(num_investments)
all_investors_df['annualised_investments_2013'] = all_investors_df['num_investments_2013'].apply(annualise)
all_investors_df.to_csv(all_investors_file, index=False)


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
all_investors_df[['percent_250m_success', 'percent_100m_success', 'percent_25m_success']] = all_investors_df.apply(
    lambda row: pd.Series(calculate_success_percentages(row['investor_uuid'], row['num_investments_2013'])),
    axis=1
)
all_investors_df.to_csv(all_investors_file, index=False)


# FEATURE 3: add investing experience in years since 2013 to each investors' record in all_investors_df

def get_first_investment_year(investor_uuid):
    # Filter coinvestor_df for rows containing the investor_uuid
    relevant_rows = coinvestor_df[coinvestor_df['investor_uuids'].str.contains(investor_uuid, na=False)]
    if not relevant_rows.empty:
        # Assuming 'founded_year' is the year column in coinvestor_df
        return relevant_rows['founded_year'].min()  # Get the earliest year
    else:
        return 2024  # No investments found, assume 0 experience (shouldnt be many investors like this)

# Apply the function to each investor_uuid in all_investors_df
all_investors_df['first_investment_year'] = all_investors_df['investor_uuid'].apply(get_first_investment_year)

# Calculate investing experience (years since the first investment)
all_investors_df['investing_experience'] = 2024 - all_investors_df['first_investment_year']

# Drop the helper column 'first_investment_year'
all_investors_df = all_investors_df.drop(columns=['first_investment_year'])
all_investors_df.to_csv(all_investors_file, index=False)

#FEATURE 4: number of categories (broad and specific) invested into by each investor

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

# Apply the functions to add the broad and specific categories to the all_investors_df
all_investors_df[['categories_broad', 'categories_broad_count']] = all_investors_df['investor_uuid'].apply(
    lambda uuid: pd.Series(get_investor_categories_broad(uuid))
)

all_investors_df[['categories_specific', 'categories_specific_count']] = all_investors_df['investor_uuid'].apply(
    lambda uuid: pd.Series(get_investor_categories_specific(uuid))
)
all_investors_df.to_csv(all_investors_file, index=False)


# Bucketing investors based on the diversity of their sector/industry investments count
all_investors_df = all_investors_df.sort_values(by=['categories_broad_count', 'categories_specific_count'], ascending=[True, True])
# Divide the DataFrame into quintiles
all_investors_df['focus_classification'] = pd.qcut(
    all_investors_df['categories_broad_count'], 
    q=5,  # Number of buckets
    labels=['Specialist', 'Focused', 'Balanced', 'Diverse', 'Universalist']  # Labels for focus_classifications
)
# Extract min and max for each focus_classification
quintile_ranges = all_investors_df.groupby('focus_classification')['categories_broad_count'].agg(['min', 'max']).reset_index()
all_investors_df.to_csv(all_investors_file, index=False)


# FEATURE 5: calculate a custom overall outlier score to bucket investors by outlier rate, taking into account 250m, 100m and 25m success, but weighted (0.85, 0.10, 0.05)
# Calculate the weighted outlier score for each investor

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

# Apply the outlier score calculation to every investor in the investors dataframe
all_investors_df['outlier_score'] = all_investors_df.apply(calculate_balanced_outlier_score, axis=1)
# Sort the DataFrame by outlier_score in ascending order
all_investors_df = all_investors_df.sort_values('outlier_score', ascending=True)
# Split into zero and nonzero outlier scores
zero_df = all_investors_df[all_investors_df['outlier_score'] == 0].copy()
nonzero_df = all_investors_df[all_investors_df['outlier_score'] > 0].copy()

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
all_investors_df.to_csv(all_investors_file, index=False)



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

bucketstats_df.to_csv(buckets_file, index=False)


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
bucketstats_df.to_csv(buckets_file, index=False)
print(bucketstats_df)


# FEATURE 8: remove all 1-investment investors from the all_investors_df as they don't have any coinvestments

# Filter out all non-L0 rows where num_investments > 1
kg_investors_df = all_investors_df[(all_investors_df['num_investments_2013'] > 1) &
                                   (all_investors_df['outlier_bucket'] != "L0")]

# Save the filtered and modified DataFrame
kg_investors_df.to_csv(kg_investors_file, index=False)