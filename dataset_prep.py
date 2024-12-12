# ------- INITIAL FILE TO RUN ON THE ORIGINAL DATASETS -----------------
# The code in this file prepares the code for the feature engineering steps in feature_eng.py

import pandas as pd

# the filepaths for the original datasets - must be in the same format as originally provided to us in the project document
orig_coinvestor = "dataset/original/Co-investor_relationships.csv"
orig_longterm = "dataset/original/Long-term_investor_performances_(2013-2022).csv"
orig_shortterm = "dataset/original/Recent_investor_performances_(2019 onwards).csv"

new_coinvestor = "dataset/coinvestor_clean.csv"
new_longterm = "dataset/long_term_clean.csv"
new_shortterm = "dataset/short_term_clean.csv"

all_investors_table = "dataset/all_investors.csv"
startups_table = "dataset/startups.csv"

# extracting the investors and creating a mapping table between uuid's and names
coinvestor_df = pd.read_csv(orig_coinvestor)
longterm_df = pd.read_csv(orig_longterm)
shortterm_df = pd.read_csv(orig_shortterm)

startupmap_df = coinvestor_df[['org_uuid', 'name']].drop_duplicates() #same thing for all startups
startupmap_df.to_csv(startups_table, index=False) #startup map table between uuid's and names

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

# Extract unique pairs of (investor_uuid, investor_name)
unique_pairs = set()
for _, row in coinvestor_df.iterrows():
    uuids = row['investor_uuids'].split(",")
    names = row['investor_names'].split(",")
    
    # Check for matching lengths
    if len(uuids) == len(names):
        for uuid, name in zip(uuids, names):
            unique_pairs.add((uuid.strip(), name.strip()))
    else:
        continue

# Create a new DataFrame from the unique pairs
all_investors_df = pd.DataFrame(list(unique_pairs), columns=['investor_uuid', 'investor_name'])

# Save the cleaned DataFrame with one UUID per row
all_investors_df.to_csv(all_investors_table, index=False)


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
