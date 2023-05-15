import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from support_functions import get_pairs, calculate_pairs_returns


# reading the data
print("Reading data file...", end ="")
tic = time.perf_counter()
path_file = "data/CRSP_stocks_daily.csv"
data = pd.read_csv(path_file)
print("done")
toc = time.perf_counter()
print(f"Reading data file took {(toc - tic):0.2f} seconds")

print("Preprocessing data...", end ="")
tic = time.perf_counter()

# coerce RET to be numeric
data.RET = pd.to_numeric(data.RET, errors = "coerce")

# format date using pandas to_datetime() method
data.DATE = pd.to_datetime(data.DATE, format = "%Y%m%d")

# select only data from 1962 onwards
data = data[data.DATE.dt.year>=1962]

# data is in long format. Will create wide matrices with volumes and prices
vol = data.pivot(index="DATE", columns="PERMNO", values="VOL")
ret = data.pivot(index="DATE", columns="PERMNO", values="RET")

# remove possible nonsensical returns
ret[ret<-1] = -1
ret.head()

# unique dates
dates = vol.index
total_days = len(dates)

print("done")
toc = time.perf_counter()
print(f"Preprocessing data took {(toc - tic):0.2f} seconds")


# initialization to backtest strategy
n_formation = 12
n_trading = 6
num_pairs = 20
d_open = 2
wait1d = 1

# storage for results
strat_returns_cc_w1d = pd.DataFrame(np.zeros((total_days, n_trading)), index = dates, columns = ["P_"+str(i+1) for i in range(n_trading)])
strat_returns_fi_w1d = strat_returns_cc_w1d.copy()
num_open_pairs_w1d = pd.DataFrame(np.zeros((total_days, n_trading)), index = dates, columns = ["P_"+str(i+1) for i in range(n_trading)])

# create indices of months in sample
month_id = pd.Series(dates.month)
month_id = (month_id.diff()!=0)
month_id[0] = 0
month_id = month_id.cumsum()
unique_months = month_id.unique()

import time

for i_port in range(n_trading):

    port_name = "P_" + str(i_port + 1)

    print("Running portfolio " + str(i_port + 1) + " of ", str(n_trading) + "...", end ="")
    tic = time.perf_counter()

    # Each portfolio pairs can start after (n_formation + i - 1) months
    # eg. portfolio 1 can start after 12 months if n_formation = 12
    #     portfolio 2 can start after 13 months etc

    for i in np.arange(start=n_formation + i_port, stop=len(unique_months) - n_trading + 1, step=n_trading):
        # tic = time.perf_counter()
        train = np.array(unique_months[i - n_formation:i])
        test = np.array(unique_months[i:i + n_trading])
        form_dates = pd.date_range(dates[month_id == train.min()][0], dates[month_id == train.max()][-1])
        trade_dates = pd.date_range(dates[month_id == test.min()][0], dates[month_id == test.max()][-1])

        # print("Formation: ", form_dates[0], " to ", form_dates[-1])
        # print("Trading: ", trade_dates[0], " to ", trade_dates[-1])

        # check available stocks

        # select only stocks:
        #  - with returns for entire formation period
        #  - with volumes > 0 for every day of formation period

        form_ret = ret[form_dates[0]:form_dates[-1]].copy()

        # daily volumes for formation period
        form_vol = vol[form_dates[0]:form_dates[-1]].copy()
        form_vol = form_vol.fillna(0)

        # toc = time.perf_counter()
        # print(f"Slicing data took {(toc - tic):0.4f} seconds")

        # tic = time.perf_counter()
        # boolean to identify eligible stocks
        ava_stocks = (form_ret.isna().sum() == 0) & ((form_vol == 0).sum() == 0)

        # formation and trading returns for selected stocks
        form_ret = ret.loc[form_dates[0]:form_dates[-1], ava_stocks]
        trade_ret = ret.loc[trade_dates[0]:trade_dates[-1], ava_stocks]

        # select pairs
        pairs = get_pairs(form_ret, 20)

        # toc = time.perf_counter()
        # print(f"Selecting pairs took {(toc - tic):0.4f} seconds")

        # tic = time.perf_counter()
        # trade pairs
        trades = calculate_pairs_returns(trade_ret, pairs, d_open, wait1d)
        # toc = time.perf_counter()
        # print(f"Calculating pairs returns took {(toc - tic):0.4f} seconds")

        # store results
        strat_returns_cc_w1d.loc[trade_dates[0]:trade_dates[-1], port_name] = trades["returns_cc"].values
        strat_returns_fi_w1d.loc[trade_dates[0]:trade_dates[-1], port_name] = trades["returns_fi"].values
        num_open_pairs_w1d.loc[trade_dates[0]:trade_dates[-1], port_name] = (trades["directions"] != 0).sum(
            axis=1).values

    toc = time.perf_counter()
    print("done")
    print(f"Running this portfolio took {(toc - tic) / 60.:0.2f} minutes")

# returns on portfolio are the averages of six overlapping portfolios
ret_cc_w1d = strat_returns_cc_w1d.mean(axis = 1)
ret_fi_w1d = strat_returns_fi_w1d.mean(axis = 1)

# dataframe with daily returns of committed capital (cc) and fully invested (fi)
ret_daily = pd.DataFrame({"ret_cc": ret_cc_w1d, "ret_fi": ret_fi_w1d})

# calculate monthly returns - used resample at monthly frequency and lambda function to compound
ret_monthly = ret_daily.resample('M').agg(lambda x: (x + 1).prod() - 1)

# save results
ret_daily.to_csv("results/daily_returns.csv")
ret_monthly.to_csv("results/monthly_returns.csv")
