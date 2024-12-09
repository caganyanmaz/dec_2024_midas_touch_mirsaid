from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import openai

load_dotenv()
openai_apikey = os.getenv("OPENAI_APIKEY")
print(openai_apikey)

REAL_WORLD_PROB = 1.9 #1.9 percent chance of randomly picking a 'successful' startup, where successful is defined as an outlier ($250M+ raised)

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None) # Show full content of each column

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

# extracting the investors and creating a mapping table between uuid's and names
'''
investormap_df = shortterm_df[['investor_uuid', 'investor_name']].drop_duplicates()
longterm_investors = longterm_df[['investor_uuid', 'investor_name']].drop_duplicates()
new_investors = longterm_investors[~longterm_investors['investor_uuid'].isin(investormap_df['investor_uuid'])]
investormap_df = pd.concat([investormap_df, new_investors], ignore_index=True).drop_duplicates()
investormap_df.to_csv('dataset/investors.csv', index=False) # saving the created mapping table in the local dataset folder

startupmap_df = coinvestor_df[['org_uuid', 'name']].drop_duplicates() #same thing for all startups
startupmap_df.to_csv('dataset/startups.csv')
'''

# dropping the investor name column as we are now working with uuid's only (making it easier to work across data tables and one less redundant column)
'''
longterm_df = longterm_df.drop(columns=['investor_name'])
longterm_df.to_csv('dataset/long_term_clean.csv', index=False)

coinvestor_df = coinvestor_df.drop(columns=['investor_names'])
coinvestor_df.to_csv('dataset/coinvestor_clean.csv', index=False)

shortterm_df = shortterm_df.drop(columns=['investor_name'])
shortterm_df.to_csv('dataset/short_term_clean.csv', index=False)
'''

# getting rid of the startup name column in the table as I made a uuid <-> name table for mapping to and from uuid's when required
'''
coinvestor_df = coinvestor_df.drop(columns=['name'])
coinvestor_df.to_csv('dataset/coinvestor_clean.csv', index=False)
'''

#converting the percentages columns into floating point values for enabling calculations to be done on the data
'''
longterm_df['success_rate'] = longterm_df['success_rate'].str.replace("%", "").astype(float)
longterm_df.to_csv('dataset/long_term_clean.csv', index=False)
shortterm_df['success_rate'] = shortterm_df['success_rate'].str.replace("%", "").astype(float)
shortterm_df.to_csv('dataset/short_term_clean.csv', index=False)
'''

MEAN_PROB_SUCCESS = 2.28 #longterm_df['success_rate'].mean(), real-world chance of randomly investing in a startup that raises $250M+ over its lifetime, 2013-now
MEAN_PROB_EARLY_SUCCESS = 11.92 #shortterm_df['success_rate'].mean(), chance of randomly investing in a 2022-now founded startup that raises $25M+

#adding feature 1 to the investor table - annualised investments (how many investments a given investor makes per year)

def num_investments(investor_uuid):
    # Check if the investor_uuid is in the 'investor_uuids' column for each row, and sum the results
    return coinvestor_df['investor_uuids'].apply(lambda x: investor_uuid in x.split(',')).sum()
def annualise(num_investments): 
    #since 2013, annualise investments per year, given this dataset
    return num_investments/12 #2024 - 2013 + 1
# mapping the num_investments function to the coinvestments table to find how many of the startups each of the investors had invested in (since 2013, as that is the oldest data)
# annualising the investments made per year since the dataset initial year, 2013 to 2024
'''
investors_df['num_investments_2013'] = investors_df['investor_uuid'].apply(num_investments)
investors_df['annualised_investments_2013'] = investors_df['num_investments_2013'].apply(annualise)
investors_df.to_csv("dataset/investors.csv")
'''

#adding feature 2 to investor table - 250M+ outlier rate since 2013, 100M+ outlier rate since 2019, 25M+ outlier rate since 2022
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

'''
# Apply the function to each investor_uuid and unpack the results into new columns
investors_df[['percent_250m_success', 'percent_100m_success', 'percent_25m_success']] = investors_df.apply(
    lambda row: pd.Series(calculate_success_percentages(row['investor_uuid'], row['num_investments_2013'])),
    axis=1
)
investors_df.to_csv("dataset/investors.csv", index=False)
'''

#adding feature 3 to investor table - number of categories (broad and specific) invested into by each investor

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

'''
# Apply the functions to add the broad and specific categories to the investors_df
investors_df[['categories_broad', 'categories_broad_count']] = investors_df['investor_uuid'].apply(
    lambda uuid: pd.Series(get_investor_categories_broad(uuid))
)

investors_df[['categories_specific', 'categories_specific_count']] = investors_df['investor_uuid'].apply(
    lambda uuid: pd.Series(get_investor_categories_specific(uuid))
)
investors_df.to_csv("dataset/investors.csv", index=False)
'''


def create_startups_nocategories_df(nocategories_df, startups_df):
    # Merge the two DataFrames on 'org_uuid'
    merged_df = nocategories_df.merge(startups_df[['org_uuid', 'name']], on='org_uuid', how='inner')
    return merged_df

# Extracting the 463 startups that have null categories and removing them from the coinvestor table (463 out of 37k+ startups is not significant enough for this task)
'''
nocategories_df = coinvestor_df[coinvestor_df['category_list'].isnull()]
startupsnocategories_df = create_startups_nocategories_df(nocategories_df, startups_df)
startupsnocategories_df.to_csv("dataset/startups_nocategories.csv", index=False)
print("Mean funding per company: " + str(startupsnocategories_df['total_funding_usd'].mean()),
      "Number of companies with $250M+ raised: " + str(startupsnocategories_df['ultimate_outlier_success_250_mil_raise'].sum()),
      "Number of companies with $100M+ raised since 2019: " + str(startupsnocategories_df['interim_success_100_mil_founded_year_2019_or_above'].sum()),
      "Number of companies with $25M+ raised since 2022: " + str(startupsnocategories_df['recent_success_25_mil_raise_founded_year_2022_or_above'].sum()), sep="\n")

Mean funding per company: 4459869.515555556
Number of companies with $250M+ raised: 0
Number of companies with $100M+ raised since 2019: 1
Number of companies with $25M+ raised since 2022: 3
'''

# Bucketing investors based on the diversity of their sector/industry investments count
'''
investors_df = investors_df.sort_values(by=['categories_broad_count', 'categories_specific_count'], ascending=[True, True])
# Divide the DataFrame into quintiles
investors_df['quintile'] = pd.qcut(
    investors_df['categories_broad_count'], 
    q=5,  # Number of buckets
    labels=['Specialist', 'Focused', 'Balanced', 'Diverse', 'Universalist']  # Labels for quintiles
)
# Extract min and max for each quintile
quintile_ranges = investors_df.groupby('quintile')['categories_broad_count'].agg(['min', 'max']).reset_index()
print(quintile_ranges)
#        quintile  min  max
# 0    Specialist    5   16
# 1       Focused   17   21
# 2      Balanced   22   26
# 3       Diverse   27   34
# 4  Universalist   35   49
investors_df.to_csv("dataset/investors.csv", index=False)
'''

