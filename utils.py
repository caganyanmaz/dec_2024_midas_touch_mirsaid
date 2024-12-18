import pandas as pd
from constants import *


def pd_set_show_full_content():
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_colwidth', None) # Show full content of each column


def serialize_investor_group(investor_uuids):
    investors = investor_uuids.split(',')
    investors_clean = list(map(lambda x: x.strip(), investors))
    return ','.join(investors_clean)


def get_investors(investment):
    for investor_raw in investment.investor_uuids.split(','):
        investor = investor_raw.strip()
        yield investor


def get_investor_columns(investors_df):
    investor_columns = {}
    for i in investors_df.index:
        investor_uuid = investors_df.at[i, 'investor_uuid']
        investor_columns[investor_uuid] = i
    return investor_columns


def is_25m(investment):
    return investment.total_funding_usd >= 25000000 and investment.founded_year >= YEAR_THRESHOLDS['25m']

def is_100m(investment):
    return investment.total_funding_usd >= 100000000 and investment.founded_year >= YEAR_THRESHOLDS['100m']

def is_250m(investment):
    return investment.total_funding_usd >= 250000000 and investment.founded_year >= YEAR_THRESHOLDS['250m']

def calculate_list_success_rates(investments):
    counters = {
        'total': 0,
        '25m': 0,
        '100m': 0,
        '250m': 0
    }
    for investment in investments:
        if is_25m(investment):
            counters['25m'] += 1
        if is_100m(investment):
            counters['100m'] += 1
        if is_250m(investment):
            counters['250m'] += 1
        counters['total'] += 1
    return counters['25m'] / counters['total'], counters['100m'] / counters['total'], counters['250m'] / counters['total']
