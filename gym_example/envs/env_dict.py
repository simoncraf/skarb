from gym import spaces
from gym.envs.registration import EnvSpec
from gym.envs.registration import register
import gym
import numpy as np
import pandas as pd
import random
import pickle
import unittest

import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import convert_to_base_env
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.evaluate import rollout
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.test_utils import check
from torch import negative

url = 'https://raw.githubusercontent.com/simoncraf/skarb/main/stocks/smi.csv'
df = pd.read_csv(url)
df = df.set_index('Date')
df = df.drop(columns = 'PGHN.SW')
euro = np.ones(df.shape[0])
df['EURO'] = euro


class PortfolioManagementDict(gym.Env):    
    states_elements = 3
    states = ['prices','weights','pf_value']

    PRICES = 0
    WEIGHTS = 1
    PF_VALUE = 2
    MAX_STEPS = df.shape[0]
    WINDOW_LOCAL_MAX = 50

    transaction_cost = 0.5
    window = 100
    rebalance = 1
    
    

    def __init__(self, data = df, capital = 10000, max_volatility = -0.3, evaluate=False):
        self.data = data
        self.initial_capital = capital
        self.max_volatility = max_volatility
        self.pf_value = capital
        self.pf_total_values = []
        for _ in range(self.WINDOW_LOCAL_MAX):
            self.pf_total_values.append(0)
        self.pf_total_values.pop()
        self.pf_total_values.insert(0,self.pf_value)
        self.max_value = self.initial_capital
        self.max_drawdown = 0
        self.action_space = gym.spaces.Box(
            low = 0.0,
            high = 1.0,
            shape=(self.data.shape[1],),
            dtype=np.float32,
            )
        '''
        self.observation_space = gym.spaces.Box(
            low = 0.0,
            high= np.inf,
            shape=(self.window,self.data.shape[1]), 
            dtype=np.float32
        )
        '''
        self.evaluate = evaluate
        self.max_idx = self.data.shape[0]
        
        self.observation_space =  gym.spaces.Dict(
            {
                "prices":gym.spaces.Box(low = 0.0,high=np.inf, shape=(self.window,self.data.shape[1]), dtype=np.float32),
                "weights":gym.spaces.Box(low = 0.00, high = 1.00, shape=(self.data.shape[1],),dtype=np.float32),
                "volatilities":gym.spaces.Box(low = -np.inf, high = np.inf, shape=(3,),dtype=np.float32),
            }
        )

        
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
        returns = returns * action_weights
        returns = sum(returns)
        return returns

    def get_sortino():
        pass

    def step(self, action):
        self.last_action = action
        prices = self.state['prices']
        weights = self.state['weights']
        volatilities = self.state['volatilities']
        pf_value = self.pf_value
        index = self.index
        rebalance = self.rebalance
        done = self.done

        val_min = self.initial_capital + (self.initial_capital * self.max_volatility)

        new_weights = self._eval_action(action)
        
        changes = 0
        for i  in range(len(new_weights)):
            if(new_weights[i] != weights[i]):
                changes += 1

        cost = changes * PortfolioManagementDict.transaction_cost
        
        returns = self._get_returns(index,rebalance,new_weights)
        new_pf_value = pf_value + (pf_value * returns)  - cost
        self.pf_total_values.pop()
        self.pf_total_values.insert(0,new_pf_value)

        if(new_pf_value > self.max_value):
            self.max_value = new_pf_value

        volatilities_init = (new_pf_value/self.initial_capital) - 1
        volatilities_init = round(volatilities_init,7)
        volatilities[0] = volatilities_init

        local_max = max(self.pf_total_values)
        volatilities_local = (new_pf_value/local_max) - 1
        if(volatilities_local > volatilities[2]):
            volatilities_local -= volatilities[2] #Es igual a la diferencia amb la volatilitat max
            volatilities_local = round(volatilities_local,7)

        volatilities[1] = volatilities_local

        final_returns = (new_pf_value / pf_value) - 1

        index += rebalance
        if(index >= self.data.shape[0]):
            done = True
        '''
        if(min(volatilities_init,volatilities_local) < self.max_volatility):
            done = True
        '''
        if(done):
            new_prices_window = prices
            reward = 0
        else:
            reward = final_returns + (volatilities_init+volatilities_local)/2
            reward = round(reward,4)
            new_prices_window = self.data.iloc[index-self.window:index].values

        self.pf_value = new_pf_value
        state = {'prices':new_prices_window,'weights':new_weights,'volatilities':volatilities}

        self.state = state
        self.done = done
        self.index = index

        return state, reward, done, {'pf_value':pf_value,'val_min':val_min,'final_weights':new_weights} 

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
        volatilities = np.array([0,0,self.max_volatility])
        self.state = {'prices':prices_window,'weights':initial_weights,'volatilities':volatilities}

        return self.state

    def render(self, mode="human"):
        pass


#select_env = "PortfolioManagementDict-v1"
#select_env = "fail-v1"
#register_env(select_env, lambda config: PortfolioManagementDict())

env = PortfolioManagementDict()
for _ in range(1):
    state = env.reset()
    sum_reward = 0
    n_step = 10000
    pf_value = []
    index_value = 10000
    index_values = []
    states = []
    
    for step in range(n_step):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        pf_value.append(info['pf_value'])
        val_min = info['val_min']
        env.render()
        if(done==1):
            print('Done')
            break
        #print("action: ",action," cumulative reward: ", round(sum_reward),3)
            
