import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def avg_rk_by_decile(c_flat: pd.Series, fc_flat: pd.Series):
    try:
        deciles_c = pd.qcut(c_flat, q=10, labels=False, duplicates="raise")
    except:
        return pd.Series(np.ones(10)*np.nan)
    avg_rk = fc_flat.rank(pct=True).groupby(deciles_c).mean()
    return avg_rk

def compute_lreg_metrics(c_flat: np.ndarray, hc_flat: np.ndarray, fc_flat: np.ndarray):
    scaler = StandardScaler()
    c_flat = scaler.fit_transform(c_flat.reshape(-1, 1))
    hc_flat = scaler.fit_transform(hc_flat.reshape(-1, 1))
    fc_flat = scaler.fit_transform(fc_flat.reshape(-1, 1))
    X_1 = sm.add_constant(hc_flat)
    model_1 = sm.OLS(fc_flat, X_1).fit()
    X_12 = sm.add_constant(np.hstack((hc_flat, c_flat)))
    model_12 = sm.OLS(fc_flat, X_12).fit()
    r2_1 = model_1.rsquared
    r2_12 = model_12.rsquared
    f_test = model_12.compare_f_test(model_1)
    p_value = f_test[1]
    std_res_1 = np.std(model_1.resid)
    std_res_12 = np.std(model_12.resid)
    return r2_1 / r2_12, std_res_1 / std_res_12, p_value

def compute_metrics(c_flat, hc_flat, fc_flat):
    assert len(c_flat) == len(fc_flat)
    if len(fc_flat) == 0:
        nan_avg_rk = pd.Series(np.ones(10)*np.nan)
        return nan_avg_rk, np.nan, np.nan, np.nan, np.nan, np.nan, nan_avg_rk
    if hc_flat is not None:
        assert len(c_flat) == len(hc_flat)
        r2_ratio, std_res_ratio, p_value = compute_lreg_metrics(c_flat, hc_flat, fc_flat)
        hc_flat = pd.Series(hc_flat)
    else:
        r2_ratio, std_res_ratio, p_value = np.nan, np.nan, np.nan
    c_flat = pd.Series(c_flat)
    fc_flat = pd.Series(fc_flat)
    avg_rk = avg_rk_by_decile(c_flat, fc_flat)
    spearman = c_flat.rank().corr(fc_flat.rank())
    spearman_h = hc_flat.rank().corr(fc_flat.rank()) if hc_flat is not None else np.nan
    avg_rk_h = avg_rk_by_decile(hc_flat, fc_flat) if hc_flat is not None else np.nan
    return avg_rk, spearman, r2_ratio, std_res_ratio, p_value, spearman_h, avg_rk_h

def avg_rk_mse(ser: pd.Series):
    ref = np.linspace(0.05, 0.95, 10)
    return np.mean((ser.to_numpy() - ref)**2)

def filter_top_hst_correl(hst_corr: pd.DataFrame, r=0.25, return_indices=True):
    """
    Output: 1d boolean mask (n_pairs,)
    """
    if not (0 < r < 1): raise ValueError(f"r must be between 0 and 1: {r}")
    i_indices, j_indices = np.triu_indices(hst_corr.shape[0], k=1)
    hst_corr_flat = hst_corr.to_numpy()[i_indices, j_indices]
    threshold = np.percentile(hst_corr_flat, (1 - r) * 100)
    mask = hst_corr_flat >= threshold
    if return_indices:
        return mask, i_indices, j_indices
    return mask

def get_top(ser: pd.Series, r):
    q = ser.quantile(q=(1-r))
    mask = ser >= q
    return mask

def filter_top_hst_correl_ptck(hst_corr: pd.DataFrame, r=0.25, symmetrization="and", return_indices=True):
    """
    Output: 1d boolean mask (n_pairs,)
    """
    if not (0 < r < 1): raise ValueError(f"r must be between 0 and 1: {r}")
    np.fill_diagonal(hst_corr.values, np.nan)
    mask_df = hst_corr.apply(get_top, r=r, axis=0)
    if symmetrization == "and":
        mask_df = mask_df & mask_df.T
    elif symmetrization == "or":
        mask_df = mask_df | mask_df.T
    i_indices, j_indices = np.triu_indices(mask_df.shape[0], k=1)
    mask = mask_df.to_numpy()[i_indices, j_indices]
    if return_indices:
        return mask, i_indices, j_indices
    return mask