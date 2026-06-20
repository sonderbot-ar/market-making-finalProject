# market-making-finalProject
## Market-Making 
### Backtesting Framework 
A reproducible quantitative trading project for developing, backtesting, and evaluating market-making algorithms using historical market data, with experiment tracking via MLflow and automated CI/CD workflows for deployment and analysis.

June 19 (Mark):
* Added .csv in .gitignore so that the repo won't recieve the datasets directly.
* Datesets are stored here https://drive.google.com/drive/folders/1xbz5ldyhyo2olywCMJ0_rt2-val1m2_F?usp=sharing instead.
* Used updated versions of kaggle and pandas (kaggle 2.2.2 and pandas 3.0.3, see requirements.txt)
* Added a data scraping file called datesets/kaggle_scrape.py to get a little of bit fresher data. Recommended BTC date points:
    for training:
        2024-06-14 Post-ETF stable liquidity, ideal “clean market making environment”
        2024-03-05 ETF hype + strong inflows + upward trend pressure
        2024-08-05 Risk-off macro + crypto liquidation cascade behavior
    for validation:
        2025-02-20 Clean regime shift test
        2025-04-15 Slight expansion phase, tests parameter sensitivity
    for testing:
        2025-06-12 ETF-driven stability period
        2025-07-10 Tests robustness under tight spreads + sudden moves

Note: kaggle_commands.txt gets outdated data, can be deleted now.

June 20 (Mark):
* Created a preprocessing file called src/data/data_preprocessor.py to add columns such as "mid_price", "price_diff", "log_return", "rolling_stdev", "sigma", "delta_ask", "delta_bid", "kappa", "A" to prepare the datasets for Inventory Management Market Making Models like Avellaneda-Stoikov.
* In gitignore, I added .csv and copy.py so that any dataset or backup files wouldn't be pushed to github.
* In src/, created 3 files:
    avellaneda_stoikov.py - runs the mathematical model against the dataset (initialized inventory = 1 BTC) and outputs in src/results/
    trained_model.py - uses a trained machine learning model of XGBoost to do the same thing as avellaneda-stoikov and saves in src/results
    backtest_version1.py - loads the outputs from src/results, checks against original datasets and validates the outputs. It also calculates the final PnL, inventory variance, and trade counts at the end of the simulation.

Latest results were:
==============================
 RESULTS: AVELLANEDA-STOIKOV
 =============================
 Total Fills     : 972
 Max Inv Exposure: 5.20 BTC
 Final PnL       : $-9251.84
 Sharpe Ratio    : -26.03
==============================

==============================
 RESULTS: XGBOOST ML
 =============================
 Total Fills     : 1175
 Max Inv Exposure: 69.30 BTC
 Final PnL       : $105667.60
 Sharpe Ratio    : 21.65
==============================
