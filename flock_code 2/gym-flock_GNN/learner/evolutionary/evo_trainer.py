import torch
import numpy as np
from .population import Population
from .fitness import evaluate_fitness
from learner.actor import Actor


def mutate(actor, sigma):
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(sigma * torch.randn_like(p))


def train_evolutionary(env, args, device):

    pop_size = args.getint('pop_size', 20)
    elite_frac = args.getfloat('elite_frac', 0.2)
    sigma = args.getfloat('mutation_sigma', 0.05)
    generations = args.getint('generations', 50)
    episode_len = args.getint('episode_len', 100)

    (obs_state, obs_gso) = env.reset()
    n_s = obs_state.shape[1]
    n_a = env.nu

    hidden_layers = eval(args.get('hidden_layers'))
    k = args.getint('k')
    ind_agg = args.getint('ind_agg')

    base_actor = Actor(n_s, n_a, hidden_layers, k, ind_agg).to(device)

    population = Population(base_actor, pop_size)

    elite_k = max(1,int(pop_size*elite_frac))

    history = []

    for gen in range(generations):

        for ind in population.individuals:
            ind.fitness = evaluate_fitness(env, ind.actor, device, episode_len)

        population.sort()
        elites = population.elites(elite_k)

        best_fitness = elites[0].fitness
        history.append(best_fitness)

        print(f"[GEN {gen}] Best fitness = {best_fitness:.4f}")

        new_inds = [elite.clone() for elite in elites]

        while len(new_inds) < pop_size:
            parent = np.random.choice(elites).clone()
            mutate(parent.actor, sigma)
            new_inds.append(parent)

        population.individuals = new_inds

    stats = {
        'mean': np.mean(history[-10:]),
        'std': np.std(history[-10:]),
        'history': history
    }
    return stats