from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock
import torch
import sys

from learner.gnn_cloning import train_cloning
from learner.gnn_dagger import train_dagger
from learner.gnn_baseline import train_baseline


def run_experiment(args):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)
    env.env.params_from_cfg(args)

    # print(gym.__)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize params tuple
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    alg = args.get('alg').lower()
    if alg == 'dagger':
        stats, train_eval_time, train_eval_rew = train_dagger(env, args, device)
    elif alg == 'cloning':
        stats = train_cloning(env, args, device)
    elif alg == 'baseline':
        stats = train_baseline(env, args)
    else:
        raise Exception('Invalid algorithm/mode name')
    return stats, train_eval_time, train_eval_rew


def main():
    fname = "cfg/dagger_leader.cfg"
    #help help

    config_file = path.join(path.dirname(__file__), fname)
    print(config_file)
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False

    if config.sections():
        for section_name in config.sections():
            if not printed_header:
                print(config[section_name].get('header'))
                printed_header = True
            stats = run_experiment(config[section_name])
            print(section_name + ", " + str(stats['mean']) + ", " + str(stats['std']))
    else:
        val, train_eval_time, train_eval_rew = run_experiment(config[config.default_section])
        print(val)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(train_eval_time, train_eval_rew)
    np.save('train_eval_time.npy', train_eval_time)
    np.save('train_eval_rew.npy', train_eval_rew)
    plt.xlabel('episodes')
    plt.ylabel('cost')
    plt.savefig(f'train_error_process.png')



if __name__ == "__main__":
    main()
