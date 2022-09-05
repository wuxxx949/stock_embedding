import time
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_sp500_list() -> Dict[str, Dict[str, str]]:
    """fetch raw sp500 company from wiki

    Returns:
        List[str]: _description_
    """
    out = {}
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    gics_mapping = make_gics_mapping()
    for _, row in df.iterrows():
        ticker = row['Symbol'].replace('.', '-')
        out[ticker] = {'proper_name': row['Security'],
                       'GICS_sector': row['GICS Sector'],
                       'GICS_group': gics_mapping[row['GICS Sub-Industry']]['industry_group'],
                       'GICS_industry': gics_mapping[row['GICS Sub-Industry']]['industry'],
                       'GICS_subindustry': row['GICS Sub-Industry']}

    return out

def make_gics_mapping() -> Dict[str, Dict[str, str]]:
    df = pd.read_html('https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard')[0]
    df = df.loc[:, df.columns.str.endswith('.1')]
    out = {}
    for _, row in df.iterrows():
        tmp = {}
        tmp['industry'] = row[2]
        tmp['industry_group'] = row[1]
        tmp['sector'] = row[0]
        out[row[3]] = tmp

    return out

def sleep_time(loc: float, scale: float):
    out = 0
    while out <= 1:
        out = np.random.normal(loc, scale)

    return out


def fetch_daily_price(tickers: List[str],
                      start: str,
                      end: str,
                      batch_size: int=20) -> pd.DataFrame:
    if len(tickers) % batch_size == 1:
        batch_size += 1

    df_lst = []
    n = -(-len(tickers) // batch_size)
    for i in range(n):
        # print(tickers[i * batch_size: i * batch_size + batch_size])
        ticker_str = ' '.join(tickers[i * batch_size: i * batch_size + batch_size])
        df = yf.download(ticker_str, start=start, end=end)
        df = df.loc[:, df.columns.get_level_values(0) == 'Adj Close']
        df.columns = df.columns.droplevel(0)
        df_lst.append(df.dropna(how='all').reset_index().melt(id_vars='Date', var_name='ticker'))
        time.sleep(sleep_time(3, 1))

    price_df = pd.concat(df_lst).reset_index(drop=True)
    price_df.columns = ['date', 'ticker', 'adj_price']

    return price_df


def daily_return_calc(price_df: pd.DataFrame) -> pd.DataFrame:
    """generate daily return from adjusted price

    Args:
        price_df (pd.DataFrame): daily adjusted price

    Returns:
        pd.DataFrame: daily return with column
    """
    price_df['shifted_price'] = price_df.sort_values(['ticker', 'date']).groupby('ticker')['adj_price'].shift()
    price_df['daily_return'] = np.log(price_df['adj_price']) - np.log(price_df['shifted_price'])

    return price_df.dropna(subset=['daily_return'])


if __name__ == '__main__':
    make_gics_mapping()
    sp_500 = fetch_sp500_list()
    sp_500_tickers = list(sp_500.keys())
    out = fetch_daily_price(sp_500_tickers, start='2021-01-01', end='2022-08-04', batch_size=20)
    return_df = daily_return_calc(out)
