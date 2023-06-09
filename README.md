# Pairs_trading
Python implementation of the pairs trading strategy in Gatev, Goetzmann and Rouwenhorst (2006). I am not the author of that study; this is an independent replication.

This project replicates the pairs trading model of [Gatev, Goetzmann and Rouwenhorst (2006)](https://doi.org/10.1093/rfs/hhj020) in Python. I used similar pairs trading strategies in my paper [The Long and the Short of Risk Parity](https://www.pm-research.com/content/iijpormgmt/early/2022/01/26/jpm20221333), although the backtesting implementation in that paper is a bit more realistic and was done in Matlab. I had previously written an article with an R implementation, which can be found [here](https://rpubs.com/arubesam/ReplicatingGGR). 

The Jupyter notebook demonstrates the backtest of the pairs trading strategy using data donwloaded from CRSP through WRDS. The .py files are more suitable if you want to explore the code directly in Python. It contains the same functionalities but with more clear function separation.
