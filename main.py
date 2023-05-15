from support_functions import *


def main():
    # # initialization to backtest strategy
    n_formation = 12
    n_trading = 6
    num_pairs = 20
    d_open = 2
    wait1d = 1
    file = "data/CRSP_stocks_daily.csv"

    vol, ret = pre_process(file)

    # backtest strategy over entire period
    strat_returns_cc_w1d, strat_returns_fi_w1d, num_open_pairs_w1d = backtest_strategy(vol, ret, n_formation,
                                                                                       n_trading, num_pairs, d_open,
                                                                                       wait1d)
    # returns on portfolio are the averages of six overlapping portfolios
    ret_cc_w1d = strat_returns_cc_w1d.mean(axis=1)
    ret_fi_w1d = strat_returns_fi_w1d.mean(axis=1)

    # dataframe with daily returns of committed capital (cc) and fully invested (fi)
    ret_daily = pd.DataFrame({"ret_cc": ret_cc_w1d, "ret_fi": ret_fi_w1d})

    # calculate monthly returns - used resample at monthly frequency and lambda function to compound
    ret_monthly = ret_daily.resample('M').agg(lambda x: (x + 1).prod() - 1)

    # save results
    ret_daily.to_csv("results/daily_returns.csv")
    ret_monthly.to_csv("results/monthly_returns.csv")


if __name__ == "__main__":
    main()
