import numpy as np
from utils import *
from combination_methods import *
from co_feature_eng import get_investor_pairs
from scipy.sparse import coo_array
import pickle

class simple_model():
    def __init__(self, key):
        self.key = key
    def predict(self, X):
        return X[self.key]


def get_investor_pair_idx(investment):
    for investor_a, investor_b in get_investor_pairs(investment):
        yield f"{investor_a},{investor_b}"


def main():
    pd_set_show_full_content()
    input_investors_df = pd.read_csv(INPUT_INVESTORS_FILENAME)
    output_investors_df = pd.read_csv(OUTPUT_INVESTORS_FILENAME)
    input_investors_df.set_index('investor_uuid', inplace=True)
    cutoff_points = {}
    def get_investor_name(investor_uuid):
        if investor_uuid not in input_investors_df.index:
            return output_investors_df.at[investor_uuid, 'investor_name']
        return input_investors_df.at[investor_uuid, 'investor_name']

    def get_coinvestor_names(investor_uuids):
        return ', '.join([get_investor_name(investor_uuid) for investor_uuid in investor_uuids.split(',')])

    PARAMETER_TYPES = [
        (INPUT_INVESTORS_FILENAME, OUTPUT_INVESTMENTS_FILENAME, 'investor_uuid', get_investors, 'investors', get_investor_name),
        (INPUT_INVESTOR_PAIRS_FILENAME, OUTPUT_INVESTMENTS_FILENAME, 'pair', get_investor_pair_idx, 'coinvestors', get_coinvestor_names),
        (INPUT_INVESTORS_WITH_GRAPH_DATA_FILENAME, OUTPUT_INVESTMENTS_FILENAME, 'investor_uuid', get_investors, 'investors-with-graph-data', get_investor_name)
    ]
    for (input_file, output_file, index, get_indices, model_name, get_input_name) in PARAMETER_TYPES:
        best_input_indices = fully_test_model(simple_model("weighted_success_rate"), f'{model_name}-simple', input_file, output_file, index, get_indices, cutoff_points)

        with open(f'top-picks/{model_name}-simple.txt', 'w') as f:
            for idx in best_input_indices:
                name = get_input_name(idx)
                f.write(f"{name}\n")

        for model_type in ['linear', 'neural']:
            full_model_name = f'{model_name}-{model_type}'
            model = load_model(f'models/{full_model_name}.pkl')
            best_input_indicies = fully_test_model(model, full_model_name, input_file, output_file, index, get_indices, cutoff_points)

            with open(f'top-picks/{full_model_name}.txt', 'w') as f:
                for idx in best_input_indicies:
                    name = get_input_name(idx)
                    f.write(f"{name}\n")
    save_cutoff_points(cutoff_points)


def load_model(model_file):
    with open(model_file, "rb") as f:
        return pickle.load(f)


def fully_test_model(model, model_name, input_file, output_file, index, find_indices, cutoff_points):
    input_df = pd.read_csv(input_file)
    input_df.set_index(index, inplace=True)
    output_investments_df = pd.read_csv(output_file)


    for best_pick_count in [30, 100]:
        for method in [best, mean, median, mean_top, median_top]: 
            text = f"{model_name}, {method.__name__} {best_pick_count}"
            cutoff_point = test_model(model, output_investments_df, input_df, find_indices, method, text, best_pick_count)
            strat_name = f"{model_name}-{method.__name__}"
            if strat_name not in cutoff_points:
                cutoff_points[strat_name] = {}
            cutoff_points[strat_name][best_pick_count] = cutoff_point
        text = f"{model_name} LSQ {best_pick_count}"
        test_lsq_method(model, output_investments_df, input_df, find_indices, text, best_pick_count)
    return get_best_input_indices(model, input_df, 10)


def get_best_input_indices(model, input_df, pick_count):
    input_df = add_predictions_to_df(input_df, model)
    values = []
    for input in input_df.itertuples():
        values.append((input.prediction, input.Index))
    values.sort(reverse=True)
    return map(lambda x: x[1], values[:pick_count])


def test_model(model, investments_df, input_df, find_indices, combining_method, text, best_pick_count=100):
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
    cutoff_point = investment_predictions[-1][0]
    s_b, m_b, l_b, t_b = calculate_list_success_counts(best_investments)
    print_success_rates(s_b, m_b, l_b, t_b, investments_df, text)
    return cutoff_point
    #print(f"Sucess rates of all investments: {s * 100}%, {m * 100}%, {l * 100}%")
    #print(f"Success rate improvement: {s_b / s}x, {m_b / m}x, {l_b / l}x")


def test_lsq_method(model, investments_df, input_df, find_indices, text, best_pick_count=100):
    input_df = add_predictions_to_df(input_df, model)
    input_df['num_index'] = range(len(input_df))
    investments_df['num_index'] = range(len(investments_df))
    I, J, V = [], [], []
    counts = { i: 0 for i in input_df.index }
    for investment in investments_df.itertuples():
        for idx in find_indices(investment):
            if idx not in input_df.index:
                continue
            counts[idx] += 1
            I.append(input_df.at[idx, 'num_index'])
            J.append(investment.num_index)
            V.append(1)
    input_df['recent_investment_count'] = counts
    I, J, V = np.array(I), np.array(J), np.array(V)
    A = coo_array((V, (I, J)), shape=(len(input_df), len(investments_df))).toarray()
    b = (input_df['prediction']* input_df['recent_investment_count']).values
    x, residuals, rank, s = np.linalg.lstsq(A, b)
    investments_df['prediction'] = x
    best_pick_pairs = list(zip(x, investments_df.itertuples()))
    best_pick_pairs.sort(reverse=True)
    best_picks = [i for _, i in best_pick_pairs[:best_pick_count]]
    
    s_b, m_b, l_b, t_b = calculate_list_success_counts(best_picks)
    print_success_rates(s_b, m_b, l_b, t_b, investments_df, text)


def print_success_rates(s_b, m_b, l_b, t_b, investments_df, text):

    s, m, l, t = calculate_list_success_counts(investments_df.itertuples())
    s_p = s_b / t_b
    m_p = m_b / t_b
    l_p = l_b / t_b
    s_x = "{:.1f}".format(s_p * t / s) 
    m_x = "{:.1f}".format(m_p * t / m)
    l_x = "{:.1f}".format(l_p * t / l)
    s_p = "{:.1f}".format(s_p * 100)
    m_p = "{:.1f}".format(m_p * 100)
    l_p = "{:.1f}".format(l_p * 100)
    s_r = "{:.1f}".format(s_b * 100 / s)
    m_r = "{:.1f}".format(m_b * 100 / m)
    l_r = "{:.1f}".format(l_b * 100 / l)
    print("{:<50}".format(f"{text}: "), end="")
    print(f"({s_p}%, {m_p}%, {l_p}%), ({s_r}%, {m_r}%, {l_r}%), ({s_b}, {m_b}, {l_b}), ({s_x}x, {m_x}x, {l_x}x)")


def add_predictions_to_df(input_df, model):
    if 'prediction' in input_df.columns:
        return input_df
    input_df = input_df.select_dtypes(include=['float64', 'int64']) 
    input_df.fillna(0, inplace=True)
    input_df['prediction'] = model.predict(input_df)
    return input_df


def save_cutoff_points(cutoff_points):
    if len(cutoff_points) == 0:
        return
    column_names = list((list(cutoff_points.values())[0]).keys())
    column_names = ["strat_name"] + column_names
    cutoff_points_df = pd.DataFrame(columns=column_names)
    for strat_name, values in cutoff_points.items():
        row = [strat_name] + list(values.values())
        cutoff_points_df.loc[len(cutoff_points_df)] = row
    cutoff_points_df.to_csv(CUTOFF_POINTS_FILENAME, index=False)


if __name__ == "__main__":
    main()