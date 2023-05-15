from sklearn.metrics import pairwise_distances
import itertools
import pandas as pd
import numpy as np


# write a function to select pairs
# function takes as input an array of formation returns and the desired number of pairs
# obs: form_ret column names are inherited original ret object
def get_pairs(form_ret, n_pairs):
    # replace first return with 0 so cum returns start at 1
    form_ret.iloc[0, :] = 0

    # create object with cumulative returns
    P = (form_ret + 1.).cumprod()

    # define number of stocks and total number of pairs
    num_stocks = P.shape[1]
    total_pairs = int(P.shape[1] * (P.shape[1] - 1) / 2)

    # create column with pair combinations to calculate distances
    pairs = np.asarray(list(itertools.permutations(P.columns, 2)))

    # keep only one set of combinations
    pairs = pd.DataFrame(pairs[pairs[:, 1] > pairs[:, 0]], columns=["leg_1", "leg_2"])

    # calculate distances between normalized prices
    pairs_dist = pairwise_distances(P.transpose(), P.transpose())
    pairs_dist = pd.Series(pairs_dist[np.triu_indices(num_stocks, k=1)])
    pairs["dist"] = pairs_dist

    # remove pairs with 0 distance if any
    pairs = pairs[pairs.dist > 0]

    # order according to distance and select pairs
    pairs = pairs.sort_values("dist", ascending=True)
    pairs = pairs.loc[pairs.index[0:min(n_pairs, pairs.shape[0])]]

    # for these pairs, store the standard deviation of the spread
    pairs["spread_std"] = np.std(np.asarray(P.loc[:, pairs.leg_1]) - np.asarray(P.loc[:, pairs.leg_2]), axis=0, ddof=1)

    pairs.index = np.arange(pairs.shape[0])
    # returns selected pairs
    return pairs


# function to calculate returns on a set of pairs over a trading period

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
        if (~np.isnan(t_open)):
            while (~np.isnan(t_open) & (t_open < last_day - wait1d)):
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
