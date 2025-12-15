from gym.envs.registration import register

register(
    id='FlockingRelative-v0',
    entry_point='gym_flock.envs.flocking:FlockingRelativeEnv',
    max_episode_steps=1000,
)

register(
    id='FlockingLeader-v0',
    entry_point='gym_flock.envs.flocking:FlockingLeaderEnv',
    max_episode_steps=200,
)