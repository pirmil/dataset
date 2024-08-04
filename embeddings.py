import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import os

OCC_PTCK = 'occ_ptck'

def get_embeddings1d_da(embeddings1d: pd.DataFrame, n_features: int, weights_col=''):
    """
    Input: (index=nothing; columns=clean_id_qis, date, features, [opt] weights_col)
    Output: (index=nothing; columns=clean_id_qis, date, features)
    """
    add_col = 2 if (weights_col is None) or (len(weights_col) == 0) else 3
    assert embeddings1d.shape[1] == n_features + add_col
    unique_dates = embeddings1d['date'].unique()
    assert len(unique_dates) == 1, 'The dataframe must represent one day only!'
    date = unique_dates[0]
    embeddings1d.drop('date', axis=1, inplace=True)
    if (weights_col is None) or (len(weights_col) == 0):
        embeddings1d_da = embeddings1d.groupby('clean_id_qis').mean()
    else:
        embeddings1d_da = (embeddings1d
            .groupby('clean_id_qis')
            .apply(lambda x: (x.drop(weights_col, axis=1).multiply(x[weights_col], 0)).sum() / x[weights_col].sum(), include_groups=False))
    embeddings1d_da['date'] = date
    embeddings1d_da.reset_index(drop=False, inplace=True)
    assert embeddings1d.shape[1] == n_features + 2
    return embeddings1d_da

def get_embeddings_da(embeddings1d_das: List[pd.DataFrame]):
    """
    Input: List of embeddings1d_da and each of them has (index=nothing; columns=clean_id_qis, date, features)
    Return: (MultiIndex=clean_id_qis, date; columns=features)
    """
    return pd.concat(embeddings1d_das, axis=0).set_index(['clean_id_qis', 'date'])

def get_embeddings(embeddings_da: pd.DataFrame, window='7D', min_periods=1):
    """
    Input: (MultiIndex=clean_id_qis, date; columns=features)
    Output: (MultiIndex=date, clean_id_qis; columns=features)
    """
    return (embeddings_da
    .sort_index(level=['clean_id_qis', 'date'])
    .reset_index(level='clean_id_qis', drop=False)
    .groupby('clean_id_qis')
    .rolling(window, min_periods=min_periods)
    .mean()
    .swaplevel(i='clean_id_qis', j='date')
    .sort_index(level='date')
)

def get_embeddings1d_da_noda(embeddings1d: pd.DataFrame, n_features: int):
    """
    Input: (index=nothing; columns=clean_id_qis, date, features, [opt] weights_col)
    Output: (index=nothing; columns=clean_id_qis, date, features, occ_ptck)
    """
    assert embeddings1d.shape[1] == n_features + 2
    unique_dates = embeddings1d['date'].unique()
    assert len(unique_dates) == 1, 'The dataframe must represent one day only!'
    date = unique_dates[0]
    embeddings1d.drop('date', axis=1, inplace=True)
    embeddings1d_by_id_qi = embeddings1d.groupby('clean_id_qis')
    embeddings1d_da = embeddings1d_by_id_qi.mean()
    embeddings1d_da[OCC_PTCK] = embeddings1d_by_id_qi.size()
    embeddings1d_da.reset_index(drop=False, inplace=True)
    embeddings1d_da['date'] = date
    assert embeddings1d_da.shape[1] == n_features + 3
    return embeddings1d_da

def get_embeddings_da_noda(embeddings1d_das: List[pd.DataFrame], n_features=384):
    """
    Input: List of embeddings1d_da s.t. (index=nothing; columns=clean_id_qis, date, features, occ_ptck)
    Output: (MultiIndex=clean_id_qis, date; columns=features, occ_ptck)
    """
    
    embeddings_da = pd.concat(embeddings1d_das, axis=0).set_index(['clean_id_qis', 'date'])
    assert embeddings_da.shape[1] == n_features + 1
    return embeddings_da

def get_embeddings_noda(embeddings_da: pd.DataFrame, window='7D', min_periods=1, n_features=384, min_rolling_occ=0):
    """
    Input: (MultiIndex=clean_id_qis, date; columns=features)
    Output: (MultiIndex=date, clean_id_qis; columns=features)
    """
    return (embeddings_da
        .sort_index(level=['clean_id_qis', 'date'])
        .reset_index(level='clean_id_qis', drop=False)
        .groupby("clean_id_qis")
        .apply(compute_rolling_weighted_mean, weights_col=OCC_PTCK, n_features=n_features, min_rolling_occ=min_rolling_occ, mp=min_periods, w=window, include_groups=False)
        .swaplevel(i='clean_id_qis', j='date')
        .sort_index(level='date')
    )


def compute_rolling_weighted_mean(df: pd.DataFrame, weights_col: str=OCC_PTCK, n_features: int=384, w: int='7D', mp: int=1, min_rolling_occ=1):
    """
    Input: (index=Datetime; columns=features, weights_col)
    Output: (index=Datetime; columns=features)
    """
    assert df.shape[1] == n_features + 1
    assert df.shape[0] == df.index.nunique()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    weights = df[weights_col]
    df.drop(weights_col, axis=1, inplace=True)
    rolling_weighted_sum = (df
        .mul(weights, axis=0)
        .rolling(window=w, min_periods=mp)
        .sum())
    weights_rolling_sum = (weights
        .rolling(window=w, min_periods=mp)
        .sum())
    mask_occ = weights_rolling_sum >= min_rolling_occ
    weights_rolling_sum = weights_rolling_sum[mask_occ]
    rolling_weighted_sum = rolling_weighted_sum[mask_occ]
    rolling_weighted_mean = rolling_weighted_sum.div(weights_rolling_sum, axis=0)
    return rolling_weighted_mean

def to_cos_matrix(emb: pd.DataFrame):
    return pd.DataFrame(cosine_similarity(emb.to_numpy()), index=emb.index, columns=emb.index)