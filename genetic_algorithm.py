import math
import individual as individual
import numpy as np
from deap import base
from copy import deepcopy
from deap.tools.emo import sortNondominated as sort_nondominated
from deap import creator
import random
from projectq.ops import (CNOT, CX, Rx,
                          Ry, Rz)
from constants import N_LAYERS, N_QUBITS, POP_SIZE, ESL, crossover_rate, cross_parents, tournament_size, num_select


creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax, model_params=None)


def mutate_ind(individual):
    mutation_choice_fn = random.choice([
        individual.discrete_uniform_mutation,
        individual.continuous_uniform_mutation,
        individual.sequence_insertion,
        individual.sequence_and_inverse_insertion,
        individual.insert_mutate_invert,
        individual.sequence_deletion,
        individual.sequence_replacement,
        individual.sequence_swap,
        individual.sequence_scramble,
        individual.permutation_mutation,
        individual.move_gate
    ])
    mutation_choice_fn()
    return individual


class Evolution:
    def __int__(self, connectivity="ALL"):
        self.ESL = ESL
        self.connectivity = connectivity
        self.nlayers = N_LAYERS
        self.allowed_gates = [Rx, Ry, Rz, CNOT]
        self.POP_SIZE = POP_SIZE

    def generate_random_circuit_layer(self, initialize=True):
        if initialize:
            p = 1 / 10
        else:
            p = 1 / self.ESL
        # cir_length = np.random.geometric(p)
        cir_length = 5
        produced_circuit = []
        for i in range(N_QUBITS):
            # produced_circuit.append(("SFG", H, i))
            produced_circuit.append(("SG", Rz, i, 0.5))
            produced_circuit.append(("SG", Ry, i, 0.5))
            produced_circuit.append(("SG", Rz, i, 0.5))
        for i in range(cir_length):
            gate = CX
            control, target = random.sample(range(N_QUBITS), 2)
            produced_circuit.append(("TFG", gate, control, target))
        return produced_circuit

    def cnot_layer(self, initialize):
        combinations = []
        if initialize:
            p = 1 / 1.5
            combinations = [(control, N_QUBITS-1) for control in range(N_QUBITS - 1)]
        else:
            p = 1 / ESL
        n_cnot = np.random.geometric(p)
        for _ in range(n_cnot):
            control, target = random.sample(range(N_QUBITS), 2)
            combinations.append((control, target))
        return combinations

    def new_pop(self, pop_size=None):
        if pop_size is None:
            Popsize = POP_SIZE
        else:
            Popsize = pop_size
        pop = []
        c_layers = []
        for _ in range(Popsize):
            layers = []
            for _ in range(N_LAYERS):
                produced_circuit = self.generate_random_circuit_layer(initialize=True)
                layers.append(produced_circuit)
            c_layers.append(layers)
            ind = creator.Individual(layers, N_QUBITS)
            ind.model_params = None
            pop.append(ind)
        return pop

    def mate(self, parent1, parent2):
        return parent1.cross_over(parent2), parent2.cross_over(parent1)  # Individual

    def tournament_selection(self, pop, fitness_of_pop, tournament_size, num_select):
        """
        :param pop:
        :param fitness_of_pop:
        :param tournament_size: 同小组竞赛的数量
        :param num_select: 选择环节胜出的个体数
        :return:
        """
        selected_individuals = []
        selected_ind_fit = []
        remaining_pop = list(zip(pop, fitness_of_pop))

        for _ in range(num_select):
            tournament = random.sample(remaining_pop, tournament_size)
            tournament.sort(key=lambda x: x[1])
            winner = tournament[0][0]
            winner_fitness = tournament[0][1]
            selected_individuals.append(winner)
            selected_ind_fit.append(winner_fitness)
            # 从剩余种群中移除已经选择的个体
            remaining_pop.remove((winner, winner_fitness))
        return selected_individuals, selected_ind_fit

    def crossover(self, pop, crossover, toolbox):
        mate_individual = []
        num_cross = int(crossover)
        crossover_pop = deepcopy(pop)
        parents = crossover_pop[:cross_parents]
        for _ in range(num_cross):  # 对整个种群执行交叉操作
            parent1, parent2 = random.sample(parents, 2)
            individuals1 = individual.Individual(parent1.clayers, N_QUBITS)
            individuals2 = individual.Individual(parent2.clayers, N_QUBITS)
            child1, child2 = self.mate(individuals1, individuals2)  # 执行交叉操作  # individual class

            mate_individual.extend([creator.Individual(ind.clayers, N_QUBITS) for ind in child1])
            mate_individual.extend([creator.Individual(ind.clayers, N_QUBITS) for ind in child2])
        return mate_individual  # creator class

    def mutate_individuals(self, ranks, N, toolbox, current_rank=1):
        L = len(ranks)  # creator class
        T = 0
        # Calculate the summation of exponential terms
        for i in range(L):
            T += math.exp(-current_rank - i)
            T += math.exp(-current_rank - i)

        cps = []

        for _ in range(N):
            random_number = random.uniform(0, T)

            list_index = -1
            right_border = 0
            for i in range(L):
                right_border += math.exp(-current_rank - i)
                if random_number <= right_border:
                    list_index = i
                    break
            if list_index == -1:
                list_index = L - 1
            left_border = right_border - math.exp(-current_rank - list_index)
            element_index = math.floor(
                len(ranks[list_index]) * (random_number - left_border) / (right_border - left_border))

            while len(ranks[list_index]) == 0:
                list_index += 1
                if len(ranks[list_index]) != 0:
                    element_index = random.choice(range(len(ranks[list_index])))

            if element_index >= len(ranks[list_index]):
                element_index = -1

            cp = deepcopy(ranks[list_index][element_index])
            individual = cp
            cp_mutat = toolbox.mutate_ind(individual)

            cp_class = creator.Individual(cp_mutat.clayers, N_QUBITS)
            cps.append(cp_class)
        return cps

    def select_and_evolve(self, pop, toolbox):
        ranks = sort_nondominated(pop, len(pop))
        to_carry = len(ranks[0])
        recall_of_pop = [ind.fitness.values[0] for ind in pop]
        print(f'before: select fitness_of_pop\n {recall_of_pop}\n')
        individuals, individuals_fit = self.tournament_selection(pop, recall_of_pop, tournament_size, num_select)
        next_generation = []
        crossover = int(len(pop) * crossover_rate)
        N = len(pop) - to_carry - crossover
        N = N if N > 0 else 1
        mutated_individuals = self.mutate_individuals(ranks, N, toolbox, current_rank=1)  # creator class
        next_generation.extend(mutated_individuals)
        mate_individuals = self.crossover(pop, crossover, toolbox)  # creator class
        next_generation.extend(mate_individuals)
        for ind in next_generation:
            if not hasattr(ind, 'model_params'):
                chosen_ind = random.choice(individuals)
                if hasattr(chosen_ind, 'model_params'):

                    ind.model_params = chosen_ind.model_params
                else:
                    ind.model_params = pop[0].model_params
        print('next_generation', len(next_generation))
        return individuals, next_generation
