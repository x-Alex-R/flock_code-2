import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv
from matplotlib.animation import FuncAnimation


class FlockingLeaderEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv, self).__init__()
        self.n_leaders = 3
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver_leaders = None
        self.quiver_others = None
        self.half_leaders = int(self.n_leaders / 2.0)

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        # 重写step方法
        assert u.shape == (self.n_agents, self.nu)
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u

        # x, y position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5 * self.mask
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5 * self.mask
        # x, y velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt * self.mask
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt * self.mask

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingLeaderEnv, self).reset()
        # leader的运动速度随机生成
        self.x[0:self.n_leaders, 2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max, high=self.v_max, size=(1, 1)) # 速度范围-5~5
        return (self.state_values, self.state_network)

    def render(self, mode='human', key=(0, 0)):
        super(FlockingLeaderEnv, self).render(mode, key)
        
        X1 = self.x[0:self.n_leaders, 0]
        Y1 = self.x[0:self.n_leaders, 1]
        U1 = self.x[0:self.n_leaders, 2]
        V1 = self.x[0:self.n_leaders, 3]

        X2 = self.x[self.n_leaders:, 0]
        Y2 = self.x[self.n_leaders:, 1]
        U2 = self.x[self.n_leaders:, 2]
        V2 = self.x[self.n_leaders:, 3]

        if self.quiver_leaders == None:
            self.quiver_leaders = self.ax.quiver(X1, Y1, U1, V1, color='red', scale=10, scale_units='inches', width=0.005, headwidth=4)
            self.quiver_others = self.ax.quiver(X2, Y2, U2, V2, color='black', scale=10, scale_units='inches', width=0.004, headwidth=3)
        else:
            self.quiver_leaders.set_offsets(self.x[0:self.n_leaders, 0:2])
            self.quiver_leaders.set_UVC(U1, V1)
            self.quiver_others.set_offsets(self.x[self.n_leaders:, 0:2])
            self.quiver_others.set_UVC(U2, V2)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if key[1]%10 == 0:
            self.fig.savefig("figures/Episode{}_{}".format(key[0], key[1]))