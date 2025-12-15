import torch
import numpy as np
from collections import deque

@torch.no_grad()
def evaluate_fitness(env, actor, device, episode_len):

    obs = env.reset()
    state, gso = obs

    # state: (N, F)
    # gso:   (N, N) or (K, N, N) depending on env

    N, F = state.shape
    K = actor.k

    # === 初始化 K 步 buffer ===
    state_buffer = deque(maxlen=K)
    gso_buffer   = deque(maxlen=K)


    for _ in range(K):
        state_buffer.append(state)
        gso_buffer.append(gso)

    total_cost = 0.0

    for _ in range(episode_len):


        # delay_state: (1, K, F, N)
        delay_state = torch.tensor(
            np.stack(state_buffer, axis=0),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0).permute(0, 1, 3, 2)

        # delay_gso: (1, K, N, N)
        delay_gso = torch.tensor(
            np.stack(gso_buffer, axis=0),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        action = actor(delay_state, delay_gso)

        action = action.squeeze(0).squeeze(0)

        action = action.permute(1, 0)

        assert action.shape == (env.n_agents, env.nu), action.shape

        action = action.cpu().numpy()


        (state, gso), cost, _, _ = env.step(action)
        total_cost += cost


        state_buffer.append(state)
        gso_buffer.append(gso)

    return -total_cost