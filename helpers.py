import numpy as np
import pandas as pd
import pickle
import os
from typing import List


def mkdir(path, verbose=True):
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose: print(f"Path created at {path}")

def save_pickle(obj, path, file, verbose=True):
    mkdir(path, verbose)
    filename = f'{path}/{file}'
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
    if verbose: print(f'Saved file at {filename}')

def load_pickle(path, file, verbose=False):
    filename = f'{path}/{file}'
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    if verbose: print(f"Loaded from {filename}")
    return obj

def sort_dates_as_str(dates: List):
    dates = pd.to_datetime(dates).sort_values().strftime('%Y-%m-%d').to_list()

def convert_level0dates_to_str(df: pd.DataFrame):
    try:
        df.index = df.index.set_levels(df.index.get_level_values(0).unique().strftime('%Y-%m-%d'), level=0)
        return df
    except:
        return df

def flatten_square_df(df: pd.DataFrame, return_indices=False):
    assert df.shape[0] == df.shape[1]
    i_indices, j_indices = np.triu_indices(df.shape[0], k=1)
    df_flatten = df.to_numpy()[i_indices, j_indices]
    if return_indices:
        return df_flatten, i_indices, j_indices
    return df_flatten

def to_sorted_dates_index(df: pd.DataFrame):
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def dropnas(df: pd.DataFrame):
    df.dropna(how='all', inplace=True, axis=0)
    df.dropna(how='all', inplace=True, axis=1)
    return df