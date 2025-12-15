import torch
import copy
import random

class Individual:
    def __init__(self, actor):
        self.actor = actor
        self.fitness = None

    def clone(self):
        return copy.deepcopy(self)


class Population:
    def __init__(self, base_actor, pop_size):
        self.individuals = []
        for _ in range(pop_size):
            actor_copy = copy.deepcopy(base_actor)
            self.individuals.append(Individual(actor_copy))

    def sort(self):
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

    def elites(self, k):
        return self.individuals[:k]
