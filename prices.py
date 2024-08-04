import pandas as pd
import numpy as np


EFF_START_DATE = "2017-11-02"
EFF_END_DATE = "2021-12-02"
full_dates = pd.date_range(start=EFF_START_DATE, end=EFF_END_DATE)
prices = pd.read_parquet('prices.pq')
f_corr = pd.read_parquet('f_corr.pq')
h_corr = pd.read_parquet('h_corr.pq')
will_increase = pd.read_parquet('will_increase.pq')