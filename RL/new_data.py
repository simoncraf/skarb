from pprint import pprint
from turtle import color

from sympy import evaluate
from gym_example.envs.example_env import PortfolioManagement
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.sac as sac
import ray.rllib.agents.ddpg.apex as apex
import ray.rllib.agents.ddpg.td3 as td3
import shutil
from model import MyKerasModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



url = 'https://raw.githubusercontent.com/simoncraf/skarb/main/stocks/smi_new_data.csv'
df = pd.read_csv(url)
df = df.set_index('Date')
df = df.drop(columns = 'PGHN.SW')
euro = np.ones(df.shape[0])
df['EURO'] = euro

url2 = 'https://raw.githubusercontent.com/simoncraf/skarb/main/stocks/smi_idx_new_data.csv'
df2 = pd.read_csv(url2)
df2 = df2.set_index('Date')
df2 = df2.iloc[100: , :]
INITIAL_CAPITAL = df2.iloc[0].values
#idx_stocks = INITIAL_CAPITAL/df2.iloc[0]
#df2 = df2 * idx_stocks



select_env = "PortfolioManagement-v1"
register_env(select_env, lambda config: PortfolioManagement())
config = td3.TD3_DEFAULT_CONFIG
#config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
#agent = ppo.PPOTrainer(config, env=select_env)
agent = td3.TD3Trainer(config,env=select_env)
agent.restore("tmp/td3/checkpoint_000150/checkpoint-150")
volatilities = [-0.05,-0.07,-0.1,-0.15,-0.2,-0.35]
final_value = []
for r in range(len(volatilities)):
    env = gym.make(select_env, data=df, max_volatility = volatilities[r], capital = INITIAL_CAPITAL, evaluate = True)
    n_episodes = 15
    avg_value = []
    for _ in range(n_episodes):
        state = env.reset()
        sum_reward = 0
        n_step = 3000
        pf_value = []
        index_value = INITIAL_CAPITAL
        index_values = []
        states = []
        
        for step in range(n_step):
            if(step > 0):
                old_state = states[-1]
            action = agent.compute_single_action(state)
            state, reward, done, info = env.step(action)
            sum_reward += reward
            pf_value.append(info['pf_value'])
            val_min = info['val_min']
            if(step==0):
                old_state = state[-1]
                old_state = sum(old_state)
            values = sum(state[-1])
            returns = (values/old_state) - 1
                
            if(step>0):
                idx_returns = (index_values[-1] * returns) + index_values[-1]
            else:
                idx_returns = (INITIAL_CAPITAL * returns) + INITIAL_CAPITAL
            index_values.append(idx_returns)
            states.append(values)

            env.render()
            if(done==1):
                break
        avg_value.append(pf_value)
    final_avg_value = []

    for idx in range(len(avg_value[0])):
        final_avg_value.append(0)
        v = 0
        for i in range(n_episodes):
            v += avg_value[i][idx]
        
        final_avg_value[idx] += v
        final_avg_value[idx] /= n_episodes
    final_value.append(final_avg_value)

fig, ax = plt.subplots()
ax.set_ylim([5000, 16000])
ax.plot(final_value[0], color = 'g')
ax.plot(final_value[1], color = 'r')
ax.plot(final_value[2], color = 'y')
ax.plot(final_value[3], color = 'b')
ax.plot(final_value[4], color = 'grey')
ax.plot(final_value[5], color = 'black')
#ax.plot(avg_value[0], color = 'y', linewidth=0.5)
#ax.plot(avg_value[1], color = 'y', linewidth=0.5)
#ax.plot(avg_value[2], color = 'y', linewidth=0.5)
#ax.plot(df2, color = 'b')

for idx, val in enumerate(final_value):
    value = min(val)
    fall = (value/INITIAL_CAPITAL) - 1
    print('Val min: ',value,', fall of the ',fall,'with max volatility: ',volatilities[idx])

ax.hlines(y=(INITIAL_CAPITAL + (INITIAL_CAPITAL * volatilities[0])), xmin=0, xmax=step, linewidth=2, color='g')
ax.hlines(y=(INITIAL_CAPITAL + (INITIAL_CAPITAL * volatilities[1])), xmin=0, xmax=step, linewidth=2, color='r')
ax.hlines(y=(INITIAL_CAPITAL + (INITIAL_CAPITAL * volatilities[2])), xmin=0, xmax=step, linewidth=2, color='y')
ax.hlines(y=(INITIAL_CAPITAL + (INITIAL_CAPITAL * volatilities[3])), xmin=0, xmax=step, linewidth=2, color='b')
ax.hlines(y=(INITIAL_CAPITAL + (INITIAL_CAPITAL * volatilities[4])), xmin=0, xmax=step, linewidth=2, color='grey')
ax.hlines(y=(INITIAL_CAPITAL + (INITIAL_CAPITAL * volatilities[5])), xmin=0, xmax=step, linewidth=2, color='black')
ax.hlines(y=max(max(final_value[0]),max(final_value[1]),max(final_value[2]),max(final_value[3]),max(final_value[4])), xmin=0, xmax=step, linewidth=2, color='g')
plt.show()

