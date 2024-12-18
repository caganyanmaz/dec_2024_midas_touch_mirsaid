from utils import *
from co_feature_eng import get_investor_pairs
import pickle



def get_investor_pair_idx(investment):
    for investor_a, investor_b in get_investor_pairs(investment):
        yield f"{investor_a},{investor_b}"

def main():
    pd_set_show_full_content()
    PARAMETER_TYPES = [
        (INPUT_INVESTORS_FILENAME, OUTPUT_INVESTMENTS_FILENAME, 'investor_uuid', get_investors, 'investors'),
        (INPUT_INVESTOR_PAIRS_FILENAME, OUTPUT_INVESTMENTS_FILENAME, 'pair', get_investor_pair_idx, 'coinvestors')
    ]
    for (input_file, output_file, index, get_indices, model_name) in PARAMETER_TYPES:
        for model_file in [f'models/{model_name}-linear.pkl', f'models/{model_name}-neural.pkl']:
            fully_test_model(model_file, input_file, output_file, index, get_indices)


def fully_test_model(model_file, input_file, output_file, index, find_indices):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    input_df = pd.read_csv(input_file)
    input_df.set_index(index, inplace=True)
    output_investments_df = pd.read_csv(output_file)

    def best(predictions):
        return max(predictions)

    def mean(predictions):
        return sum(predictions) / len(predictions)
    
    def median(predictions):
        predictions.sort()
        return predictions[len(predictions) // 2]

    for method in [best, mean, median]: 
        for best_pick_count in [10, 30, 50, 100, 500, 1000, 10000]:
            print(f"Model: {model_file}, Method: {method.__name__}, Best pick count: {best_pick_count}") 
            test_model(model, output_investments_df, input_df, find_indices, method, best_pick_count)



def test_model(model, investments_df, input_df, find_indices, combining_method, best_pick_count=100):
    input_df = add_predictions_to_df(input_df, model)
    investment_predictions = []
    for investment in investments_df.itertuples():
        predictions = []
        for idx in find_indices(investment):
            if idx not in input_df.index:
                continue
            current_prediction = input_df.at[idx, 'prediction']
            predictions.append(current_prediction)
        if not predictions:
            continue
        overall_prediction = combining_method(predictions)
        investment_predictions.append((overall_prediction, investment))
    investment_predictions.sort(reverse=True)
    best_investments = []
    for _, investment in investment_predictions[:best_pick_count]:
        best_investments.append(investment)
    s_b, m_b, l_b = calculate_list_success_rates(best_investments)
    s, m, l       = calculate_list_success_rates(investments_df.itertuples())
    print(f"Success rates of best investments: {s_b * 100}%, {m_b * 100}%, {l_b * 100}%")
    print(f"Sucess rates of all investments: {s * 100}%, {m * 100}%, {l * 100}%")
    print(f"Success rate improvement: {s_b / s}x, {m_b / m}x, {l_b / l}x")



def add_predictions_to_df(input_df, model):
    if 'prediction' in input_df.columns:
        return input_df
    input_df = input_df.select_dtypes(include=['float64', 'int64']) 
    input_df.fillna(0, inplace=True)
    input_df['prediction'] = model.predict(input_df)
    return input_df


if __name__ == "__main__":
    main()