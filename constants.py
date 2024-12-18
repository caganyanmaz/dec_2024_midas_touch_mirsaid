
# Company info
ORIGINAL_FILENAME = "dataset/original.csv"

# new filepaths for cleaned data files
INPUT_INVESTMENTS_FILENAME = "dataset/training_investments.csv"
OUTPUT_INVESTMENTS_FILENAME     = "dataset/test_investments.csv"
INVESTORS_FILENAME            = "dataset/investors.csv"
STARTUPS_FILENAME             = "dataset/startups.csv"
NO_CATEGORY_STARTUP_FILENAME  = "dataset/startups_nocategories.csv"
INVESTOR_PAIR_FILENAME        = "dataset/investor_pairs.csv"

WEIGHTS = {'250m': 0.42, '100m': 0.30, '25m': 0.28} 
#WEIGHTS = {'250m': 0.34, '100m': 0.33, '25m': 0.33} 

YEAR_THRESHOLDS = {
    '250m': 2013,
    '100m': 2017,
    '25m': 2019
}

INPUT_OUTPUT_SPLIT_YEAR = 2021

INPUT_START_YEAR = 2013
INPUT_END_YEAR   = INPUT_OUTPUT_SPLIT_YEAR
CURRENT_YEAR     = 2024
