
# Company info
ORIGINAL_FILENAME = "dataset/original.csv"

# new filepaths for cleaned data files
INPUT_INVESTMENTS_FILENAME = "dataset/input_investments.csv"
OUTPUT_INVESTMENTS_FILENAME       = "dataset/output_investments.csv"

INPUT_INVESTORS_FILENAME          = "dataset/input_investors.csv"
OUTPUT_INVESTORS_FILENAME         = "dataset/output_investors.csv"
OUTPUT_INVESTORS_TEST_FILENAME    = "dataset/output_investors_test.csv"
OUTPUT_INVESTORS_TRAIN_FILENAME   = "dataset/output_investors_train.csv"

INPUT_INVESTOR_PAIRS_FILENAME        = "dataset/input_investor_pairs.csv"
OUTPUT_INVESTOR_PAIRS_FILENAME       = "dataset/output_investor_pairs.csv"
OUTPUT_INVESTOR_PAIRS_TRAIN_FILENAME = "dataset/output_investor_pairs_train.csv"
OUTPUT_INVESTOR_PAIRS_TEST_FILENAME  = "dataset/output_investor_pairs_test.csv"


INPUT_INVESTORS_WITH_GRAPH_DATA_FILENAME = "dataset/input_investors_with_graph_data.csv"



STARTUPS_FILENAME             = "dataset/startups.csv"
NO_CATEGORY_STARTUP_FILENAME  = "dataset/startups_nocategories.csv"

INVESTOR_RANK_SCORES_FILENAME = "dataset/investor_rank_scores.csv"
CENTRALITY_ANALYSIS_FILENAME = "dataset/centrality_analysis.csv"
COMMUNITY_DIRECTORY = "dataset/communities"
COMMUNITY_COMPOSITION_FILENAME = f"{COMMUNITY_DIRECTORY}/community_composition.csv"


ANN_INVESTMENT_THRESHOLD = 1.5     


CUTOFF_POINTS_FILENAME = "cutoff_points.csv"


WEIGHTS = {'250m': 250 / 400, '100m': 100 / 400, '25m': 25 / 400}
#WEIGHTS = {'250m': 0.42, '100m': 0.30, '25m': 0.28} 
#WEIGHTS = {'250m': 0.34, '100m': 0.33, '25m': 0.33} 

YEAR_THRESHOLDS = {
    '250m': 2013,
    '100m': 2017,
    '25m': 2019
}

TEST_SIZE = 0.2

INPUT_OUTPUT_SPLIT_YEAR = 2021

INPUT_START_YEAR = 2013
INPUT_END_YEAR   = INPUT_OUTPUT_SPLIT_YEAR
CURRENT_YEAR     = 2024
