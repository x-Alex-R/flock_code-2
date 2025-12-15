import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import gca

font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 14}

# TODO: add functions to change # of agents, comm radius, and initial velocity, and initial radius (and then reset)
# and adjust the initialization radius accordingly


class FlockingRelativeEnv(gym.Env):

    def __init__(self):
        self.mean_pooling = True  # normalize the adjacency matrix by the number of neighbors or not
        self.centralized = True

        # number states per agent
        self.nx_system = 4    # x,y,u,v
        # numer of observations per agent
        self.n_features = 6
        # number of actions per agent
        self.nu = 2         # a_x,a_y

        # default problem parameters
        self.n_agents = 100     # int(config['network_size'])
        self.dt = 0.01        # float(config['system_dt'])
        self.v_max = 5.0       # float(config['max_vel_init'])
        self.r_max = 1.0       #10.0  # float(config['max_rad_init']) 初始化位置时的半径范围
        # self.alpha = 0.25       # 通信率25%
        self.alpha = 1          # 通信率25%
        self.v_bias = self.v_max  # 速度偏差量

        # intitialize state matrices
        self.x = None
        self.u = None
        self.mean_vel = None
        self.init_vel = None

        self.max_accel = 1
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features), dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.seed()

    def params_from_cfg(self, args):
        self.n_agents = args.getint('n_agents')
        # r_max与粒子总数相关 self.r_max = 1*10
        self.r_max = self.r_max * np.sqrt(self.n_agents)
        self.comm_radius = np.sqrt(self.r_max * self.alpha) # float(config['comm_radius']) 势能函数临界半径  =>  粒子通信范围
        self.comm_radius2 = self.comm_radius * self.comm_radius
        
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features), dtype=np.float32)

        self.v_max = args.getfloat('v_max')
        self.v_bias = self.v_max
        self.dt = args.getfloat('dt')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # 版本v0终止条件：在游戏中走了200步
        assert u.shape == (self.n_agents, self.nu)

        self.u = u 

        # x, y position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5
        # x, y velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt

        self.compute_helpers()

        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def compute_helpers(self):
        self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape((1, self.n_agents, self.nx_system))  # 相对位置/速度
        self.r2 =  np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1], self.diff[:, :, 1]) # x^2+y^2
        np.fill_diagonal(self.r2, np.Inf)

        # 联络半径comm_radius
        if self.alpha == 1:
            self.adj_mat = np.ones((self.n_agents, self.n_agents))-np.eye(self.n_agents)
        else:
            self.adj_mat = (self.r2 < self.comm_radius2).astype(float) # 局部邻接矩阵Sn

        # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents,1)) # correct - checked this # 计算各节点出度的和
        n_neighbors[n_neighbors == 0] = 1
        self.adj_mat_mean = self.adj_mat / n_neighbors # 邻接矩阵归一化

        # 非线性预特征:公式(12)
        # (x, r_x/r^2, r_x/r^4)和(y, r_y/r^2, r_y/r^4) 共6维
        self.x_features = np.dstack((self.diff[:, :, 2], np.divide(self.diff[:, :, 0], np.multiply(self.r2, self.r2)), np.divide(self.diff[:, :, 0], self.r2),
                          self.diff[:, :, 3], np.divide(self.diff[:, :, 1], np.multiply(self.r2, self.r2)), np.divide(self.diff[:, :, 1], self.r2)))
        
        # x_features:(n_agents, n_agents, 6)
        # adj_mat:(n_agents, n_agents)
        self.state_values = np.sum(self.x_features * self.adj_mat.reshape(self.n_agents, self.n_agents, 1), axis=1)
        self.state_values = self.state_values.reshape((self.n_agents, self.n_features))

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat

    def get_stats(self):
        stats = {}

        stats['vel_diffs'] = np.sqrt(np.sum(np.power(self.x[:, 2:4] - np.mean(self.x[:, 2:4], axis=0), 2), axis=1))

        stats['min_dists'] = np.min(np.sqrt(self.r2), axis=0)
        return stats

    def instant_cost(self):  # sum of differences in velocities
        curr_variance = -1.0 * np.sum((np.var(self.x[:, 2:4], axis=0)))
        return curr_variance


    def reset(self):
        x = np.zeros((self.n_agents, self.nx_system))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.1  # 0.25
        
        # 选择合适的初始化(确保节点间相互通信)
        # generate an initial configuration with all agents connected,
        # and minimum distance between agents > min_dist_thresh
        # while语句保证无孤立的粒子且两两之间距离不太近（即不小于安全距离0.1）
        while degree < 2 or min_dist < min_dist_thresh: 
            # randomly initialize the location and velocity of all agents
            length = np.sqrt(np.random.uniform(0, self.r_max, size=(self.n_agents,)))
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
            x[:, 0] = length * np.cos(angle)
            x[:, 1] = length * np.sin(angle)

            bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
            x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0] 
            x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1] 

            # compute distances between agents
            x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
            a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
            np.fill_diagonal(a_net, np.Inf) # 找最小距离时排除节点自身

            # compute minimum distance between agents and degree of network to check if good initial configuration
            min_dist = np.sqrt(np.min(np.min(a_net)))
            a_net = a_net < self.comm_radius2
            degree = np.min(np.sum(a_net.astype(int), axis=1))

        # keep good initialization
        self.mean_vel = np.mean(x[:, 2:4], axis=0)
        self.init_vel = x[:, 2:4]
        self.x = x
        self.compute_helpers()
        return (self.state_values, self.state_network)

    def controller(self, centralized=None):
        """
        The controller for flocking from Turner 2003.
        Returns: the optimal action
        """

        if centralized is None:
            centralized = self.centralized
        
        p_sum = np.dot(self.adj_mat_mean, self.x)
        expect_v_controls = np.hstack(((p_sum[:, 2]).reshape((-1, 1)), (p_sum[:, 3]).reshape(-1, 1)))
        now_v = np.hstack(((self.x[:, 2]).reshape((-1, 1)), (self.x[:, 3]).reshape(-1, 1)))
        controls = (expect_v_controls > now_v) * 20 - 10

        return controls
       
    def render(self, mode='human', key=(0, 0)):

        #Render the environment with agents as points in 2D space
        
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[:, 0], self.x[:, 1], 'bo', markersize=4)  # Returns a tuple of line objects, thus the comma
            self.ax.plot([0], [0], 'kx')
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            plt.title('Distributed Controller')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
 