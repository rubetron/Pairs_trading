from sklearn.metrics import pairwise_distances
import itertools
import pandas as pd
import numpy as np
import time


# function to read data
def import_data(file):
    # reading the data
    print("Reading data file...", end="")
    tic = time.perf_counter()
    try:
        data = pd.read_csv(file)
    except Exception as e:
        print("An error occurred while reading the file:")
        print(e)
        return None
    print("done")
    toc = time.perf_counter()
    print(f"Reading data file took {(toc - tic):0.2f} seconds")
    return data


# function to preprocess data into wide format
def pre_process(file):
    # read data
    data = import_data(file)
    # pre-processing
    print("Preprocessing data...", end="")
    tic = time.perf_counter()
    # coerce RET to be numeric
    data.RET = pd.to_numeric(data.RET, errors="coerce")
    # format date using pandas to_datetime() method
    data.DATE = pd.to_datetime(data.DATE, format="%Y%m%d")
    # select only data from 1962 onwards
    data = data[data.DATE.dt.year >= 1962]

    # data is in long format. Will create wide matrices with volumes and prices
    vol = data.pivot(index="DATE", columns="PERMNO", values="VOL")
    ret = data.pivot(index="DATE", columns="PERMNO", values="RET")

    # remove possible nonsensical returns
    ret[ret < -1] = -1
    # unique dates
    # total_days = len(dates)

    print("done")
    toc = time.perf_counter()
    print(f"Preprocessing data took {(toc - tic):0.2f} seconds")
    return vol, ret


# write a function to select pairs
# function takes as input an array of formation returns and the desired number of pairs
# obs: form_ret column names are inherited original ret object
def get_pairs(form_ret, n_pairs):
    # replace first return with 0 so cum returns start at 1
    form_ret.iloc[0, :] = 0

    # create object with cumulative returns
    prc = (form_ret + 1.).cumprod()

    # define number of stocks and total number of pairs
    num_stocks = prc.shape[1]
    total_pairs = int(prc.shape[1] * (prc.shape[1] - 1) / 2)

    # create column with pair combinations to calculate distances
    pairs = np.asarray(list(itertools.permutations(prc.columns, 2)))

    # keep only one set of combinations
    pairs = pd.DataFrame(pairs[pairs[:, 1] > pairs[:, 0]], columns=["leg_1", "leg_2"])

    # calculate distances between normalized prices
    pairs_dist = pairwise_distances(prc.transpose(), prc.transpose())
    pairs_dist = pd.Series(pairs_dist[np.triu_indices(num_stocks, k=1)])
    pairs["dist"] = pairs_dist

    # remove pairs with 0 distance if any
    pairs = pairs[pairs.dist > 0]

    # order according to distance and select pairs
    pairs = pairs.sort_values("dist", ascending=True)
    pairs = pairs.loc[pairs.index[0:min(n_pairs, pairs.shape[0])]]

    # for these pairs, store the standard deviation of the spread
    pairs["spread_std"] = np.std(np.asarray(prc.loc[:, pairs.leg_1]) - np.asarray(prc.loc[:, pairs.leg_2]), axis=0,
                                 ddof=1)

    pairs.index = np.arange(pairs.shape[0])
    # returns selected pairs
    return pairs


# function to calculate returns on a set of pairs over a given trading period
def calculate_pairs_returns(trade_ret, pairs, d_open=2, wait1d=1):
    # trade_ret : array of returns over trading period
    # pairs  : data frame with information about pairs
    # d_open : number of standard deviations to open a pair
    # wait1d : number of days to wait before opening trade, default = 1

    # don't need dates as indices; switch to integers
    trade_ret_dates = trade_ret.index
    trade_ret.index = np.arange(trade_ret.shape[0])
    trade_ret.iloc[0, :] = 0

    # to treat delisting correctly, identify last non NA values
    last_valid_ret_ind = trade_ret.apply(pd.Series.last_valid_index)

    # can safely replace NAs with 0 before this index.
    for idx, column in enumerate(trade_ret.columns):
        r = trade_ret.loc[trade_ret.index[0]:last_valid_ret_ind.iloc[idx], column]
        r = r.fillna(0)
        trade_ret.loc[trade_ret.index[0]:last_valid_ret_ind.iloc[idx], column] = r

    # "normalize" trading period prices to start at 1
    trade_prc = (trade_ret + 1.).cumprod()

    # total number of days in trading period
    trading_days = trade_prc.shape[0]

    # total number of pairs considered
    num_pairs = pairs.shape[0]

    # storage for intermediate calculations
    directions = pd.DataFrame(np.zeros((trading_days, num_pairs)))
    payoffs = pd.DataFrame(np.zeros((trading_days, num_pairs)))

    # loop through pairs and check for trades
    for idx_pair, pair in pairs.iterrows():

        # create df to store weights (the w1 and w2 in GGR) and returns of legs
        pair_calcs = pd.DataFrame(np.zeros((trading_days, 9)),
                                  columns=["p_1", "p_2", "s", "direction", "w_1", "w_2", "r_1", "r_2", "payoff"])

        # identify pair legs, build prices, returns and spread vectors
        leg_1 = int(pair.leg_1)
        leg_2 = int(pair.leg_2)
        pair_calcs.p_1 = trade_prc.loc[:, leg_1]
        pair_calcs.p_2 = trade_prc.loc[:, leg_2]
        pair_calcs.r_1 = trade_ret.loc[:, leg_1]
        pair_calcs.r_2 = trade_ret.loc[:, leg_2]
        last_day = max(pair_calcs.p_1.last_valid_index(), pair_calcs.p_2.last_valid_index())
        pair_calcs.r_1 = pair_calcs.r_1.fillna(0)
        pair_calcs.r_2 = pair_calcs.r_2.fillna(0)

        pair_calcs.s = (pair_calcs.p_1 - pair_calcs.p_2) / pair.spread_std

        open_ids = np.array(trade_ret.index * (np.abs(pair_calcs.s) > d_open))
        open_ids = open_ids[open_ids != 0]
        open_ids = open_ids[open_ids <= last_day]

        close_ids = np.array(trade_ret.index[np.sign(pair_calcs.s).diff() != 0])
        close_ids = close_ids[~np.isnan(close_ids)]
        close_ids = np.append(close_ids, last_day)

        # date when first trade opens
        if len(open_ids) != 0:
            t_open = open_ids[0]
        else:
            t_open = np.nan

        # if there has been a divergence in the trading period
        if ~np.isnan(t_open):
            while ~np.isnan(t_open) & (t_open < last_day - wait1d):
                # check when trade closed
                t_close = np.min(close_ids[close_ids > t_open + wait1d])

                # store direction of trade over period when trade is open
                pair_calcs.loc[(t_open + wait1d + 1): (t_close + 1), "direction"] = -np.sign(
                    pair_calcs.loc[t_open - wait1d, "s"])

                # update w1 and w2
                pair_calcs.w_1[(t_open + wait1d):(t_close + 1)] = np.append(1., (
                        1 + pair_calcs.r_1[(t_open + wait1d): (t_close)]).cumprod())
                pair_calcs.w_2[(t_open + wait1d):(t_close + 1)] = np.append(1., (
                        1 + pair_calcs.r_2[(t_open + wait1d): (t_close)]).cumprod())

                # update t_open => moves to next trade for this pair
                if any(open_ids > t_close):
                    t_open = open_ids[open_ids > t_close][0]
                else:
                    t_open = np.nan

        # calculate and store the payoffs for this pair
        pair_calcs["payoffs"] = pair_calcs.direction * (
                pair_calcs.w_1 * pair_calcs.r_1 - pair_calcs.w_2 * pair_calcs.r_2)
        payoffs.loc[:, idx_pair] = pair_calcs["payoffs"]
        directions.loc[:, idx_pair] = pair_calcs["direction"]

    directions.index = trade_ret_dates
    payoffs.index = trade_ret_dates

    # returns for committed capital approach - just the column average of payoffs
    returns_cc = payoffs.mean(axis=1)

    # for fully-invested approach, capital is divided among open pairs

    num_open_pairs = (directions != 0).sum(axis=1)
    num_open_pairs[num_open_pairs > 0] = 1. / num_open_pairs
    weights_fi = pd.concat([num_open_pairs] * num_pairs, axis=1)
    returns_fi = (weights_fi * payoffs).sum(axis=1)

    # return everything as a dictionary
    return {"pairs": pairs, "directions": directions, "payoffs": payoffs, "returns_cc": returns_cc,
            "returns_fi": returns_fi}


# function to backtest the strategy over the entire period
def backtest_strategy(vol, ret, n_formation, n_trading, num_pairs, d_open, wait1d):
    dates = vol.index
    total_days = len(dates)

    # storage for results
    strat_returns_cc_w1d = pd.DataFrame(np.zeros((total_days, n_trading)), index=dates,
                                        columns=["P_" + str(i + 1) for i in range(n_trading)])
    strat_returns_fi_w1d = strat_returns_cc_w1d.copy()
    num_open_pairs_w1d = pd.DataFrame(np.zeros((total_days, n_trading)), index=dates,
                                      columns=["P_" + str(i + 1) for i in range(n_trading)])

    # create indices of months in sample
    month_id = pd.Series(dates.month)
    month_id = (month_id.diff() != 0)
    month_id[0] = 0
    month_id = month_id.cumsum()
    unique_months = month_id.unique()

    for i_port in range(n_trading):

        port_name = "P_" + str(i_port + 1)
        print(f"Running portfolio {i_port + 1} of {n_trading}...", end="")
        tic = time.perf_counter()

        # Each portfolio pairs can start after (n_formation + i - 1) months
        # eg. portfolio 1 can start after 12 months if n_formation = 12
        #     portfolio 2 can start after 13 months etc
        date_rng = np.arange(start=n_formation + i_port, stop=len(unique_months) - n_trading + 1, step=n_trading)
        for i in date_rng:
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
            pairs = get_pairs(form_ret, num_pairs)

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
    return strat_returns_cc_w1d, strat_returns_fi_w1d, num_open_pairs_w1d
