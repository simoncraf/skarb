import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
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

class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold,
                lookback=252,
                day = 0,
                training = False):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.training = training
        self.turbulence_threshold = turbulence_threshold

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list)+2,self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.turb_arr = np.full((1,self.state_space),self.turbulence_threshold)
        self.state = np.append(self.state,self.turb_arr,axis=0)

        self.init_weights = np.full((1,self.state_space),0)
        self.init_weights[0] = 1
        self.state = np.append(self.state,self.init_weights,axis=0)
        #self.state = np.append(self.state,self.data['turbulence'].values.tolist())
        #print(self.state)
        self.terminal = False     
                
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/(self.stock_dim+1)]*(self.stock_dim+1)]
        self.date_memory=[self.data.date.unique()[0]]

        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            self.reward = 0
            plt.plot(df.daily_return.cumsum(),'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()
            
            plt.plot(self.portfolio_return_memory,'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal,{}

        else:
            weights = self.softmax_normalization(actions) 
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            percentile = stats.percentileofscore(self.df['turbulence'].values,self.data['turbulence'].values[0]) #percentil de la turbolencia
            #print('Turbulence: ',self.data['turbulence'].values[0])

            idx_turbolence = int((percentile/100) >= self.turbulence_threshold) #Veure si la turbolencia és major que el limit
            cash = weights[-1]
            cash_turb_diff = abs(cash - (percentile/100))
            self.state =  np.append(np.array(self.covs)*1000, [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.state = np.append(self.state,self.turb_arr,axis=0)
            
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights[0:-1])
            log_portfolio_return = np.log(sum((self.data.close.values / last_day_memory.close.values)*weights[0:-1]))
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)
            
            weights = np.reshape(weights[0:-1],(1,len(weights[0:-1])))
            self.state = np.append(self.state,weights,axis=0)
            # the reward is the new portfolio value or end portfolio value
            self.reward = new_portfolio_value - (new_portfolio_value * cash_turb_diff * idx_turbolence) #PF - (PF * abs(c-t) * I)
            self.reward = self.reward * self.reward_scaling
            #self.reward = cash
            #print('Reward: ',self.reward)
            #print(self.reward)
            #self.reward = new_portfolio_value * (weights[-1]/100)
            
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs)*1000, [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        #self.state = np.append(self.state,self.data['turbulence'].values.tolist())
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/(self.stock_dim+1)]*(self.stock_dim+1)]
        self.date_memory=[self.data.date.unique()[0]]

        if(self.training):
            self.turbulence_threshold = np.random.random_sample()
            self.day = random.randint(0,int(self.df.shape[0] * 0.75))
            print(self.turbulence_threshold)

        self.turb_arr = np.full((1,self.state_space),self.turbulence_threshold)
        self.state = np.append(self.state,self.turb_arr,axis=0)

        self.init_weights = np.full((1,self.state_space),0)
        self.init_weights[0] = 1
        self.state = np.append(self.state,self.init_weights,axis=0)

        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        #numerator = np.exp(actions)
        denominator = np.sum(actions)
        softmax_output = actions/denominator
        return softmax_output

    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        col_names = self.data.tic.values
        col_names = np.append(col_names,'CASH')
        df_actions.columns = col_names
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs