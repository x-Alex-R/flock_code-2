from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch
import sys

#from learner.gnn_dagger import DAGGER
from learner.state_with_delay import MultiAgentStateWithDelay

import os
 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test(args, render=True):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)
    if isinstance(env.env, gym_flock.envs.flocking.FlockingRelativeEnv):
        env.env.params_from_cfg(args)
    elif isinstance(env.env, gym_flock.envs.flocking.FlockingLeaderEnv):
        env.env.params_from_cfg(args)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_test_episodes = args.getint('n_test_episodes')

    for i in range(n_test_episodes):# 20次实验
        episode_reward = 0
        state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
        done = False
        count = 0
        while not done:
            if render:
                env.render(mode='human', key=(i, count))
            count += 1
            optimal_action = env.env.controller()
            next_state, reward, done, _ = env.step(optimal_action)
            next_state = MultiAgentStateWithDelay(device, args, next_state, prev_state=state)
            episode_reward += reward
            state = next_state
        print(episode_reward)
    env.close()


def main():
    #fname = sys.argv[1]
    fname = 'cfg/dagger_leader.cfg'
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False
    test(config[config.default_section])


if __name__ == "__main__":
    main()
