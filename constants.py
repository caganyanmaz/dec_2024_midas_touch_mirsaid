
# Company info
ORIGINAL_FILENAME = "dataset/original.csv"

# new filepaths for cleaned data files
TRAINING_INVESTMENTS_FILENAME = "dataset/training_investments.csv"
TEST_INVESTMENTS_FILENAME     = "dataset/test_investments.csv"
INVESTORS_FILENAME            = "dataset/investors.csv"
STARTUPS_FILENAME             = "dataset/startups.csv"
NO_CATEGORY_STARTUP_FILENAME  = "dataset/startups_nocategories.csv"


WEIGHTS = { 
    'weight_250m': 0.85,
    'weight_100m': 0.1,
    'weight_25m': 0.05
}

YEAR_THRESHOLDS = {
    '250m': 2013,
    '100m': 2017,
    '25m': 2019
}


TRAINING_TEST_SPLIT_YEAR = 2021

TRAINING_START_YEAR = 2013
TRAINING_END_YEAR   = TRAINING_TEST_SPLIT_YEAR

