from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cmath import inf
from turtle import done
import pandas as pd
import abc
from sqlalchemy import true
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from gym import Env
from gym.spaces import Discrete, Box
import random
from collections import Counter

import gym
import numpy as np
from hashlib import new
from gym.envs.registration import register
import random
import pandas as pd
import numpy as np
import gym.spaces
from gym.wrappers import FlattenObservation
from gym.spaces.utils import unflatten

url = 'https://raw.githubusercontent.com/simoncraf/skarb/main/stocks/smi.csv'
df = pd.read_csv(url)
df = df.set_index('Date')
df = df.drop(columns = 'PGHN.SW')
euro = np.ones(df.shape[0])
df['EURO'] = euro


class PortfolioManagement(gym.Env):    
    states_elements = 3
    states = ['prices','weights','pf_value']

    PRICES = 0
    WEIGHTS = 1
    PF_VALUE = 2
    MAX_STEPS = df.shape[0]

    transaction_cost = 0.5
    window = 100
    rebalance = 1
    
    

    def __init__(self, data = df, capital = 10000, max_volatility = -0.3, evaluate=False):
        self.data = data
        self.initial_capital = capital
        self.max_volatility = max_volatility
        self.pf_value = capital
        self.max_value = self.initial_capital
        self.max_drawdown = 0
        self.action_space = gym.spaces.Box(
            low = 0.0,
            high = 1.0,
            shape=(self.data.shape[1],),
            dtype=np.float32,
            )

        self.observation_space = gym.spaces.Box(
            low = 0.0,
            high= np.inf,
            shape=(self.window,self.data.shape[1]), 
            dtype=np.float32
        )
        self.evaluate = evaluate
        self.max_idx = self.data.shape[0]
        '''
        self.observation_space =  gym.spaces.Dict(dict(prices=gym.spaces.Box(low = 0.0,high=np.inf, shape=(self.window,self.data.shape[1]), dtype=np.float32),
                                                                weights = gym.spaces.Box(low = 0.00, high = 1.00, shape=(self.data.shape[1],),dtype=np.float32),
                                                                ))

        '''
        #self.reset()

    def _get_window(self):
        idx = random.randint(self.window,self.max_idx - self.window - 1)
        prices_window = self.data.iloc[idx-self.window:idx].values
        return prices_window, idx

    def _eval_action(self,action):
        act_weights = action
        sum_weights = sum(act_weights)
        act_weights = act_weights/sum_weights

        return act_weights

    def _get_returns(self,current_day,rebalance_day,action_weights):
        values_current_day = self.data.iloc[current_day]
        if ((current_day+rebalance_day) >= self.max_idx):
            values_rebalance_day = self.data.iloc[-1]
        else:
            values_rebalance_day = self.data.iloc[current_day + rebalance_day]
        returns = (values_rebalance_day / values_current_day) - 1
        #returns = np.append(returns,0)
        returns = returns * action_weights
        returns = sum(returns)
        return returns

    def get_sortino():
        pass

    def step(self, action):
        self.last_action = action
        prices = self.state
        #weights = self.state['weights']
        pf_value = self.pf_value
        index = self.index
        rebalance = self.rebalance
        done = self.done

        val_min = self.initial_capital + (self.initial_capital * self.max_volatility)

        new_weights = self._eval_action(action)
        '''
        changes = 0
        for i  in range(len(new_weights)):
            if(new_weights[i] != weights[i]):
                changes += 1

        cost = changes * PortfolioManagement.transaction_cost
        '''
        returns = self._get_returns(index,rebalance,new_weights)
        new_pf_value = pf_value + (pf_value * returns) # - cost
        final_returns = (new_pf_value / pf_value) - 1

        index += rebalance
        if(index >= self.data.shape[0]):
            done = True

        if(done):
            new_prices_window = prices
        else:
            new_prices_window = self.data.iloc[index-self.window:index].values

        if((new_pf_value/self.initial_capital) - 1 < self.max_volatility):
            reward = 0
        else:
            reward = abs(final_returns - self.max_volatility)

        self.pf_value = new_pf_value
        state = new_prices_window

        self.state = state
        self.done = done
        self.index = index

        return state, reward, done, {'pf_value':pf_value,'val_min':val_min} 

    def reset(self):
        prices_window, self.index = self._get_window()
        if(self.evaluate):
            self.index = self.window
            prices_window = self.data.iloc[self.index-self.window:self.index].values
        initial_weights = np.full((self.data.shape[1],), 0)
        initial_weights[-1] = 1
        self.done = False
        if(self.evaluate == False):
            self.max_volatility = round(random.uniform(0.05, 0.35),3) * -1
        self.max_value = self.initial_capital
        self.max_drawdown = 0
        self.pf_value = self.initial_capital
        self.state = prices_window

        return self.state

    def render(self, mode="human"):
        pass