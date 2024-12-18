from utils import *
import pickle


def main():
    pd_set_show_full_content()
    with open("models/linear.pkl", "rb") as f:
        model = pickle.load(f)
    input_df = pd.read_csv(INPUT_INVESTORS_FILENAME)
    input_df.set_index('investor_uuid', inplace=True)
    output_investments_df = pd.read_csv(OUTPUT_INVESTMENTS_FILENAME)
    test_model(model, output_investments_df, input_df, get_investors, 100)


def test_model(model, investments_df, input_df, find_indices, best_pick_count=100):
    input_df = add_predictions_to_df(input_df, model)
    investment_predictions = []
    for investment in investments_df.itertuples():
        best_prediction = 0
        for idx in find_indices(investment):
            if idx not in input_df.index:
                continue
            current_prediction = input_df.at[idx, 'prediction']
            best_prediction = max(best_prediction, current_prediction)
        investment_predictions.append((best_prediction, investment))
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