import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv
from gym import spaces
import matplotlib.pyplot as plt
# from matplotlib.pyplot import gca
import os


class FlockingLeaderEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv, self).__init__()
        self.n_leaders = 2
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver = None
        self.half_leaders = int(self.n_leaders / 2.0)

    def params_from_cfg(self, args):
        self.alg = args.get('alg')
        
        if self.alg == 'Centralized':
            self.comm_radius = float('inf')
        elif self.alg == 'Distributed':
            self.comm_radius = args.getfloat('comm_radius')
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)

        self.n_agents = args.getint('n_agents')
        self.r_max = self.r_max * np.sqrt(self.n_agents)

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.v_max = args.getfloat('v_max')
        self.v_bias = self.v_max
        self.max_accel = args.getint('max_accel')
        self.dt = args.getfloat('dt')
        
        self.n_leaders = args.getint('n_leaders')
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel) # crutial
        self.u = u
        # print(u)

        # x, y position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5 * self.mask
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5 * self.mask
        # x, y velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt * self.mask
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt * self.mask

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def controller(self):
        vel_mean = np.dot(self.adj_mat_mean, self.x[:, 2:4])
        vel_curr = self.x[:, 2:4]
        controls = (vel_mean - vel_curr) * 100 # acceleration based on difference between mean v and current v
        # print(controls)
        
        return controls
    
    def get_stats(self):

        stats = {}

        stats['vel_diffs'] = np.sqrt(np.sum(np.power(self.x[:, 2:4] - np.mean(self.x[:, 2:4], axis=0), 2), axis=1))

        stats['min_dists'] = np.min(np.sqrt(self.r2), axis=0)
        
        stats['vel_mse'] = np.power(self.x[:, 2:4] - self.x[0, 2:4], 2).mean() # mse with leader's velocity
        
        return stats

    def reset(self):
        super(FlockingLeaderEnv, self).reset()
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max,
                                                                                         high=self.v_max, size=(1, 1))
        return (self.state_values, self.state_network)

    def render(self, mode='human', episode=0, curr_step=0):
        """
        Render the environment with agents as points in 2D space
        """
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[:self.n_leaders, 0], self.x[:self.n_leaders, 1],
                                  'go')  # Returns a tuple of line objects, thus the comma
            line2, = self.ax.plot(self.x[self.n_leaders:, 0], self.x[self.n_leaders:, 1],
                                  'bo')  # Returns a tuple of line objects, thus the comma
            self.ax.plot([0], [0], 'kx')
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            # a = gca()
            # a.set_xticklabels(a.get_xticks(), font)
            # a.set_yticklabels(a.get_yticks(), font)
            plt.title(self.alg + ' Controller')
            self.fig = fig
            self.line1 = line1
            self.line2 = line2

        # leaders
        self.line1.set_xdata(self.x[:self.n_leaders, 0])
        self.line1.set_ydata(self.x[:self.n_leaders, 1])
        # followers
        self.line2.set_xdata(self.x[self.n_leaders:, 0])
        self.line2.set_ydata(self.x[self.n_leaders:, 1])
        
        X = self.x[0:self.n_leaders, 0]
        Y = self.x[0:self.n_leaders, 1]
        U = self.x[0:self.n_leaders, 2]
        V = self.x[0:self.n_leaders, 3]

        if self.quiver == None:
            self.quiver = self.ax.quiver(X, Y, U, V, color='r')
        else:
            self.quiver.set_offsets(self.x[0:self.n_leaders, 0:2])
            self.quiver.set_UVC(U, V)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # save img
        img_dir = f'figures/{self.alg}'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        if curr_step % 10 == 0:
            self.fig.savefig(f'{img_dir}/Episode{episode}_{curr_step}')
