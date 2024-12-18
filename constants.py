
# Company info
ORIGINAL_FILENAME = "dataset/original.csv"

# new filepaths for cleaned data files
INPUT_INVESTMENTS_FILENAME = "dataset/training_investments.csv"
OUTPUT_INVESTMENTS_FILENAME     = "dataset/test_investments.csv"
INVESTORS_FILENAME            = "dataset/investors.csv"
STARTUPS_FILENAME             = "dataset/startups.csv"
NO_CATEGORY_STARTUP_FILENAME  = "dataset/startups_nocategories.csv"


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


def is_25m(investment):
    return investment.total_funding_usd >= 25000000 and investment.founded_year >= YEAR_THRESHOLDS['25m']

def is_100m(investment):
    return investment.total_funding_usd >= 100000000 and investment.founded_year >= YEAR_THRESHOLDS['100m']

def is_250m(investment):
    return investment.total_funding_usd >= 250000000 and investment.founded_year >= YEAR_THRESHOLDS['250m']