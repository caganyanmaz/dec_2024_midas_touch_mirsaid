import pandas as pd


def pd_set_show_full_content():
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None) # Show full content of each column


def serialize_investor_group(investor_uuids):
    investors = investor_uuids.split(',')
    investors_clean = list(map(lambda x: x.strip(), investors))
    return ','.join(investors_clean)


def get_investors(investment, investor_columns=None):
    for investor_raw in investment.investor_uuids.split(','):
        investor = investor_raw.strip()
        if investor_columns is None:
            yield investor
            continue
        if investor in investor_columns:
            yield investor_columns[investor]


def get_investor_columns(investors_df):
    investor_columns = {}
    for i in investors_df.index:
        investor_uuid = investors_df.at[i, 'investor_uuid']
        investor_columns[investor_uuid] = i
    return investor_columns

