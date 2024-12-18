# ------- INITIAL FILE TO RUN ON THE ORIGINAL DATASETS -----------------
# The code in this file prepares the code for the feature engineering steps in feature_eng.py

import pandas as pd
from constants import *



def split_df(df, test_size=0.2, random_state=None):
    train_df, output_investments_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, output_investments_df


def main():
    original_df = pd.read_csv(ORIGINAL_FILENAME)
    startupmap_df = original_df[['org_uuid', 'name']].drop_duplicates() 
    startupmap_df.to_csv(STARTUPS_FILENAME, index=False) #startup map table between uuid's and names

    original_df = extract_null_category_startups(original_df, startupmap_df)
    extract_investor_data(original_df)

    original_df = original_df.drop(columns=['investor_names'])
    original_df = original_df.drop(columns=['name'])
    input_investments_df, output_investments_df = split_df(original_df, 'founded_year', INPUT_OUTPUT_SPLIT_YEAR)


    input_investments_df.to_csv(INPUT_INVESTMENTS_FILENAME)
    output_investments_df.to_csv(OUTPUT_INVESTMENTS_FILENAME)



def create_startups_nocategories_df(nocategories_df, startups_df):
    # Merge the two df's on 'org_uuid'
    merged_df = nocategories_df.merge(startups_df[['org_uuid', 'name']], on='org_uuid', how='inner')
    return merged_df


def extract_null_category_startups(original_df, startupmap_df):
    nocategories_df = original_df[original_df['category_list'].isnull()]
    startupsnocategories_df = create_startups_nocategories_df(nocategories_df, startupmap_df)
    startupsnocategories_df.to_csv(NO_CATEGORY_STARTUP_FILENAME, index=False)
    nocategories_org_uuids = nocategories_df['org_uuid'].unique()

    original_df = original_df[~original_df['org_uuid'].isin(nocategories_org_uuids)]
    return original_df


def extract_investor_data(original_df):
    unique_pairs = set()
    for _, row in original_df.iterrows():
        uuids = row['investor_uuids'].split(",")
        names = row['investor_names'].split(",")
        
        # Check for matching lengths
        if len(uuids) == len(names):
            for uuid, name in zip(uuids, names):
                unique_pairs.add((uuid.strip(), name.strip()))
        else:
            continue
    investors_df = pd.DataFrame(list(unique_pairs), columns=['investor_uuid', 'investor_name'])
    investors_df.to_csv(INPUT_INVESTORS_FILENAME, index=False)
    investors_df.to_csv(OUTPUT_INVESTORS_FILENAME, index=False)


def split_df(df, col, val):
    condition = df[col] <= val
    return df[condition], df[~condition]




if __name__ == "__main__":
    main()
