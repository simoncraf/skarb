import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import PPO,SAC,TD3
matplotlib.use('Agg')
import datetime

from finrl import config
from finrl import config_tickers
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.finrl_meta.data_processor import DataProcessor
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
import sys
import random
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import stats
from environment import StockPortfolioEnv
import os

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

BONDS = ['^IRX','^FVX','^TNX','^TYX']
ETF = ['CAPD','QLD','EZA','KBA','FBZ','IEUS']

tickers = config_tickers.DOW_30_TICKER

INITIAL_DATE = '2020-01-01'
FINAL_DATE = '2022-06-10'
'''
df = YahooDownloader(start_date = INITIAL_DATE,
                     end_date = FINAL_DATE,
                     ticker_list = tickers).fetch_data()

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=True,
                    user_defined_feature = False)

df = fe.preprocess_data(df)

print(df.tail())
df.to_csv('./dow30.csv')
'''

df = pd.read_csv('./dow30.csv')
print(df)
df_original = df.copy()

# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  return_list.append(return_lookback)

  covs = return_lookback.cov().values 
  cov_list.append(covs)

  
df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)

if('turbulence' not in config.INDICATORS):
    config.INDICATORS.append('turbulence')
config.INDICATORS

df_tech = df[config.INDICATORS]
df_tech = (df_tech - df_tech.expanding().min())/(df_tech.expanding().max() - df_tech.expanding().min())
df_tech = df_tech.fillna(0.5)

for col in df_tech.columns:
    df[col] = df_tech[col]

from scipy import stats
trade = data_split(df,'2019-01-02', '2022-06-10')
stock_dimension = len(trade.tic.unique())
act_space = stock_dimension + 1
state_space = stock_dimension
env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "transaction_cost_pct": 0.001, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.INDICATORS,
    "turbulence_threshold":0.1, 
    "action_space": act_space, 
    "reward_scaling": 1e-4,
    "training":False
    
}

trained_ppo = PPO.load('./trained_ppo.zip')
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
df_daily_return_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo,
                        environment = e_trade_gym)
trained_td3 = TD3.load('./td3.zip')
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
df_daily_return_td3, df_actions_td3 = DRLAgent.DRL_prediction(model=trained_td3,
                        environment = e_trade_gym)

trained_sac = SAC.load('./sac.zip')
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)
df_daily_return_sac, df_actions_sac = DRLAgent.DRL_prediction(model=trained_sac,
                        environment = e_trade_gym)
