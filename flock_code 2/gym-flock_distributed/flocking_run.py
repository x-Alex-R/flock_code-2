import gym
import gym_flock
import numpy as np
import random
import configparser


def test(args, render=True):
    # initialize gym env
    env_name = args.get('env')
    env = gym.make(env_name)
    if isinstance(env.env, gym_flock.envs.flocking.FlockingLeaderEnv):
        env.params_from_cfg(args)

    # use seed
    seed = args.getint('seed')
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    episode_times = args.getint('episode_times')
    reward_sum = 0
    mse_sum = 0
    for i in range(episode_times):
        state = env.reset()
        episode_reward = 0
        done = False
        curr_step = 0
        while not done:
            action = env.controller()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render(mode='human', episode=i, curr_step=curr_step)
            curr_step += 1
        vel_mse = env.get_stats()['vel_mse']
        reward_sum += reward
        mse_sum += vel_mse

        print(f'Episode {i}, episode reward: {episode_reward}, final reward: {reward}, final velocity mse: {vel_mse}')

    print(f'Final reward mean: {reward_sum / episode_times}, final velocity mse mean: {mse_sum / episode_times}')
    
    env.close()

def main():
    config_file = 'cfg/distributed.cfg'
    config = configparser.ConfigParser()
    config.read(config_file)
    config = config["DEFAULT"]

    test(config)

if __name__ == "__main__":
    main()
