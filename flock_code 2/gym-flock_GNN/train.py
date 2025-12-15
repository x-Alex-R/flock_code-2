"""
Updated training script for GNN-based and evolutionary approaches to the flocking task.

This version of the script adds support for specifying a custom configuration
file on the command line, which allows you to run different algorithms without
modifying the source.  For example, to run the evolutionary algorithm using a
dedicated configuration you would execute:

```
python train.py cfg/evolutionary.cfg
```

When invoked without any arguments the script defaults to loading the
`cfg/dagger_leader.cfg` configuration supplied in the repository.  This file
contains multiple sections (e.g. `[DEFAULT]`, `[evolutionary]`) and the script
will iterate over all sections and report statistics for each.

The remainder of the code is based on the original `train.py` from the
repository but has been reorganised slightly for clarity.
"""

from os import path
import configparser
import numpy as np
import random
import gym
import gym_flock  # type: ignore  # ensure gym-flock is installed in your environment
import torch
import sys

from learner.gnn_cloning import train_cloning
from learner.gnn_dagger import train_dagger
from learner.gnn_baseline import train_baseline


def run_experiment(args):
    """
    Execute a single experiment based on the provided configuration section.

    The configuration dictionary should include at least the keys ``env``
    specifying the Gym environment id and ``alg`` specifying which algorithm
    to run.  Additional hyperparameters are extracted within this function.

    Returns a tuple ``(stats, train_eval_time, train_eval_rew)`` where
    ``stats`` is a dictionary containing summary statistics from the training
    procedure.  For DAGGER-based algorithms ``train_eval_time`` and
    ``train_eval_rew`` record the progress of the evaluation episodes.  For
    evolutionary runs these values will be ``None``.
    """
    # initialise gym environment
    env_name = args.get('env')
    env = gym.make(env_name)
    # propagate configuration parameters into the environment if supported
    if hasattr(env.env, 'params_from_cfg'):
        env.env.params_from_cfg(args)

    # set reproducibility seeds
    seed = args.getint('seed', fallback=0)
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # allocate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_eval_time, train_eval_rew = None, None
    alg = args.get('alg').lower()

    if alg == 'dagger':
        stats, train_eval_time, train_eval_rew = train_dagger(env, args, device)
    elif alg == 'cloning':
        stats = train_cloning(env, args, device)
    elif alg == 'baseline':
        stats = train_baseline(env, args)
    elif alg == 'evolutionary':
        # import inside function to avoid unnecessary dependency when not used
        from learner.evolutionary.evo_trainer import train_evolutionary
        stats = train_evolutionary(env, args, device)
    else:
        raise Exception(f"Invalid algorithm/mode name: {alg}")
    return stats, train_eval_time, train_eval_rew


def main():
    """Entry point for running one or more experiments from a configuration file."""
    # Determine which configuration file to load.  A filename can be provided
    # on the command line; otherwise fall back to the default DAGGER configuration.
    fname = sys.argv[1] if len(sys.argv) > 1 else "cfg/dagger_leader.cfg"
    # Resolve relative paths with respect to the script location
    config_file = path.join(path.dirname(__file__), fname)
    print(f"Loading configuration from {config_file}")
    config = configparser.ConfigParser()
    config.read(config_file)

    printed_header = False
    train_eval_time, train_eval_rew = None, None

    # When named sections are defined, run each of them; otherwise run the DEFAULT section.
    if config.sections():
        for section_name in config.sections():
            section = config[section_name]
            # If a human-readable header is provided print it once
            if not printed_header and section.get('header'):
                print(section.get('header'))
                printed_header = True
            stats, train_eval_time, train_eval_rew = run_experiment(section)
            # Print mean and std for quick comparison between algorithms
            print(f"{section_name}, {stats['mean']}, {stats['std']}")
    else:
        # No explicit sections: run the default values
        stats, train_eval_time, train_eval_rew = run_experiment(config[config.default_section])
        print(stats)

    # Optionally plot and save evaluation progress for DAGGER-based runs
    if train_eval_time is not None and train_eval_rew is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(train_eval_time, train_eval_rew)
        np.save('train_eval_time.npy', train_eval_time)
        np.save('train_eval_rew.npy', train_eval_rew)
        plt.xlabel('episodes')
        plt.ylabel('cost')
        plt.savefig('train_error_process.png')


if __name__ == "__main__":
    main()