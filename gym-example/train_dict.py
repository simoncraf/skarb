from pprint import pprint
from gym_example.envs.env_dict import PortfolioManagementDict
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


def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/dict"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "PortfolioManagementDict-v1"
    #select_env = "fail-v1"
    register_env(select_env, lambda config: PortfolioManagementDict())
    #register_env(select_env, lambda config: Fail_v1())


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    #agent = ddpg.DDPGTrainer(config,env=select_env)
    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 50

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = agent.get_policy()
    #model = policy.model
    #print(policy)


    # apply the trained policy in a rollout
    '''
    agent.restore(chkpt_file)
    env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0
    '''


if __name__ == "__main__":
    #main()
    
    select_env = "PortfolioManagement-v1"
    #select_env = "fail-v1"
    register_env(select_env, lambda config: PortfolioManagement())
    #config = ppo.DEFAULT_CONFIG.copy()
    config = td3.TD3_DEFAULT_CONFIG
    config["log_level"] = "WARN"
    #agent = ppo.PPOTrainer(config, env=select_env)
    agent = td3.TD3Trainer(config,env=select_env)
    #agent.restore("tmp/exa/checkpoint_000050/checkpoint-50")
    env = gym.make(select_env)
    for _ in range(1):
        state = env.reset()
        sum_reward = 0
        n_step = 1000
        pf_value = []
        index_value = 10000
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
            #returns = sum(returns)
            if(step>0):
                idx_returns = (index_values[-1] * returns) + index_values[-1]
            else:
                idx_returns = (10000 * returns) + 10000
            index_values.append(idx_returns)
            states.append(values)

            env.render()
            if(done==1):
                break
            #print("action: ",action," cumulative reward: ", round(sum_reward),3)
            
        fig, ax = plt.subplots()
        ax.plot(pf_value)
        ax.plot(index_values)
        ax.hlines(y=val_min, xmin=0, xmax=step, linewidth=2, color='r')
        plt.show()

    
        