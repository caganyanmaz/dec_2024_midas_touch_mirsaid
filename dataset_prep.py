# ------- INITIAL FILE TO RUN ON THE ORIGINAL DATASETS -----------------
# The code in this file prepares the code for the feature engineering steps in feature_eng.py
# This code 

import pandas as pd

# the filepaths for the original datasets - must be in the same format as originally provided to us in the project document
orig_coinvestor = "dataset/original/Co-investor_relationships.csv"
orig_longterm = "dataset/original/Long-term_investor_performances_(2013-2022).csv"
orig_shortterm = "dataset/original/Recent_investor_performances_(2019 onwards).csv"

new_coinvestor = "dataset/coinvestor_clean.csv"
new_longterm = "dataset/long_term_clean.csv"
new_shortterm = "dataset/short_term_clean.csv"
investors_table = "dataset/investors.csv"
startups_table = "dataset/startups.csv"

# extracting the investors and creating a mapping table between uuid's and names
coinvestor_df = pd.read_csv(orig_coinvestor)
longterm_df = pd.read_csv(orig_longterm)
shortterm_df = pd.read_csv(orig_shortterm)

investormap_df = shortterm_df[['investor_uuid', 'investor_name']].drop_duplicates()
longterm_investors = longterm_df[['investor_uuid', 'investor_name']].drop_duplicates()
new_investors = longterm_investors[~longterm_investors['investor_uuid'].isin(investormap_df['investor_uuid'])]
investormap_df = pd.concat([investormap_df, new_investors], ignore_index=True).drop_duplicates()
investormap_df.to_csv(investors_table, index=False) # saving the created mapping table in the local dataset folder

startupmap_df = coinvestor_df[['org_uuid', 'name']].drop_duplicates() #same thing for all startups
startupmap_df.to_csv(startups_table, index=False) #startup map table between uuid's and names, this is how you get file 

# dropping the investor name column as we are now working with uuid's only (making it easier to work across data tables and one less redundant column)
longterm_df = longterm_df.drop(columns=['investor_name'])
longterm_df.to_csv(new_longterm, index=False)

coinvestor_df = coinvestor_df.drop(columns=['investor_names'])
coinvestor_df.to_csv(new_coinvestor, index=False)

shortterm_df = shortterm_df.drop(columns=['investor_name'])
shortterm_df.to_csv(new_shortterm, index=False)

# getting rid of the startup name column in the table as I made a uuid <-> name table for mapping to and from uuid's when required
coinvestor_df = coinvestor_df.drop(columns=['name'])
coinvestor_df.to_csv(new_coinvestor, index=False)

#converting the percentages columns into floating point values for enabling calculations to be done on the data
longterm_df['success_rate'] = longterm_df['success_rate'].str.replace("%", "").astype(float)
longterm_df.to_csv(new_longterm, index=False)
shortterm_df['success_rate'] = shortterm_df['success_rate'].str.replace("%", "").astype(float)
shortterm_df.to_csv(new_shortterm, index=False)

def create_startups_nocategories_df(nocategories_df, startups_df):
    # Merge the two df's on 'org_uuid'
    merged_df = nocategories_df.merge(startups_df[['org_uuid', 'name']], on='org_uuid', how='inner')
    return merged_df

# Extracting the 463 startups that have null categories and removing them from the coinvestor table (463 out of 37k+ startups is not significant for this task, they are pretty much all not outlier success)
nocategories_df = coinvestor_df[coinvestor_df['category_list'].isnull()]
startupsnocategories_df = create_startups_nocategories_df(nocategories_df, startupmap_df)
startupsnocategories_df.to_csv("dataset/startups_nocategories.csv", index=False)
nocategories_org_uuids = nocategories_df['org_uuid'].unique()
coinvestor_df = coinvestor_df[~coinvestor_df['org_uuid'].isin(nocategories_org_uuids)]
coinvestor_df.to_csv("dataset/coinvestor_clean.csv", index=False)

# print("Mean funding per company: " + str(startupsnocategories_df['total_funding_usd'].mean()),
#       "Number of companies with $250M+ raised: " + str(startupsnocategories_df['ultimate_outlier_success_250_mil_raise'].sum()),
#       "Number of companies with $100M+ raised since 2019: " + str(startupsnocategories_df['interim_success_100_mil_founded_year_2019_or_above'].sum()),
#       "Number of companies with $25M+ raised since 2022: " + str(startupsnocategories_df['recent_success_25_mil_raise_founded_year_2022_or_above'].sum()), sep="\n")

# Mean funding per company: 4459869.515555556
# Number of companies with $250M+ raised: 0
# Number of companies with $100M+ raised since 2019: 1
# Number of companies with $25M+ raised since 2022: 3
