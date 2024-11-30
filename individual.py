import copy

import numpy.random
import random
from deap import creator, base

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

class Individual:
    """This class is a container for an individual in GA."""

    def __init__(self, clayers, args, weights=None, EMC=2.0, ESL=2.0):
        self.args = args
        self.number_of_qubits = 5
        self.clayers = clayers
        self.EMC = EMC
        self.ESL = ESL
        self.CMW = 0.3
        self.weights = weights

    def generate_random_cnots(self, initialize=False):
        if initialize:
            p = 1 / 30
        else:
            p = 1 / self.ESL
        n_cnot = numpy.random.geometric(p)
        combinations = []
        for _ in range(n_cnot):
            control, target = random.sample(range(self.number_of_qubits), 2)
            combinations.append((control, target))
        # print('调用了一次', combinations[:])
        return combinations

    def generate_cnot(self):
        control, target = random.sample(range(self.number_of_qubits), 2)
        return control, target

    def discrete_uniform_mutation(self):
        """
        随机选择一个clayer，并随机替换其中的一个门组合。
        """
        if not self.clayers:
            print(f"Exception occurred during discrete_uniform_mutation")
            return
        idx = random.randrange(len(self.clayers))
        cnot_idx = random.randrange(len(self.clayers[idx]))
        self.clayers[idx][cnot_idx] = self.generate_random_cnots()[:]

    def continuous_uniform_mutation(self):
        """
        随机选择一个clayer，并随机修改其中一个门组合的control和target位置。
        """
        if not self.clayers:
            print(f"Exception occurred during continuous_uniform_mutation")
            return
        idx = random.randrange(len(self.clayers))
        cnot_idx = random.randrange(len(self.clayers[idx]))
        control = random.randrange(self.number_of_qubits)
        target = random.randrange(self.number_of_qubits)
        # 确保control和target不相同
        while target == control:
            target = random.randrange(self.number_of_qubits)
        self.clayers[idx][cnot_idx] = (control, target)

    def sequence_insertion(self):
        """
        在随机clayer的随机位置插入一个新的门组合。
        """
        new_cnot = self.generate_cnot()
        idx = random.randrange(len(self.clayers))
        cnot_idx = random.randrange(len(self.clayers[idx]) + 1)
        # 检查新门组合是否已经存在于列表中，如果存在则不插入
        if new_cnot not in self.clayers[idx]:
            self.clayers[idx].insert(cnot_idx, new_cnot)

    def sequence_and_inverse_insertion(self):
        """
        在随机clayer的随机位置插入一个新的门组合以及其逆操作。
        """
        new_cnot = self.generate_cnot()
        idx = random.randrange(len(self.clayers))
        cnot_idx = random.randrange(len(self.clayers[idx]) + 1)
        # 检查新门组合是否已经存在于列表中，如果存在则不插入
        if new_cnot not in self.clayers[idx]:
            self.clayers[idx].insert(cnot_idx, new_cnot)
            self.clayers[idx].insert(cnot_idx + 1, (new_cnot[1], new_cnot[0]))

    def insert_mutate_invert(self):
        """
        在随机clayer的随机位置插入一个新的门组合，或者变异一个门组合，或者对一个门组合进行逆操作。
        """
        choice = random.randint(0, 2)
        if choice == 0:
            self.sequence_insertion()
        elif choice == 1:
            self.discrete_uniform_mutation()
        else:
            self.sequence_and_inverse_insertion()

    def sequence_deletion(self):
        """
        随机删除一个clayer中的一个门组合。
        """
        if not self.clayers:
            print(f"Exception occurred during sequence_deletion")
            return
        idx = random.randrange(len(self.clayers))
        del_idx = random.randrange(len(self.clayers[idx]))
        del self.clayers[idx][del_idx]

    def sequence_replacement(self):
        """
        随机选择一个clayer，并替换其中的一个门组合为一组新的门组合。
        """
        if not self.clayers:
            print(f"Exception occurred during sequence_replacement")
            return
        idx = random.randrange(len(self.clayers))
        self.clayers[idx] = self.generate_random_cnots()

    def sequence_swap(self):
        """
        随机选择两个clayer，并交换它们的位置。
        """
        if len(self.clayers) < 2:
            print(f"Exception occurred during sequence_swap")
            return
        idx1, idx2 = random.sample(range(len(self.clayers)), 2)
        self.clayers[idx1], self.clayers[idx2] = self.clayers[idx2], self.clayers[idx1]

    def sequence_scramble(self):
        """
        随机选择一部分clayer，并随机交换它们的排列顺序。
        """
        if len(self.clayers) < 2:
            return
        random.shuffle(self.clayers)

    def permutation_mutation(self):
        """
        对clayers的顺序进行变异，相当于重新排列clayers的顺序。
        """
        random.shuffle(self.clayers)

    def move_gate(self):
        if not self.clayers:
            return
        idx = random.randrange(len(self.clayers))
        new_idx = random.randrange(len(self.clayers))
        self.clayers.insert(new_idx, self.clayers.pop(idx))

    def cross_over(self, parent2):
        crossover_type = random.randint(0, 1)

        if crossover_type == 0:
            return self.crossover_layers(parent2)
        else:
            return self.crossover_within_layers(parent2)

    def crossover_layers(self, parent2):
        n_crossover_points = min(len(self.clayers), len(parent2.clayers))

        crossover_point = random.randint(0, n_crossover_points - 1)

        child1_clayers = self.clayers[:crossover_point] + parent2.clayers[crossover_point:]
        child2_clayers = parent2.clayers[:crossover_point] + self.clayers[crossover_point:]

        child1_clayers = Individual(child1_clayers, self.number_of_qubits)
        child2_clayers = Individual(child2_clayers, self.number_of_qubits)

        return child1_clayers, child2_clayers


    def crossover_within_layers(self, parent2):
        idx = random.randint(0, min(len(self.clayers), len(parent2.clayers)) - 1)
        n_crossover_operations = 2
        crossover_points = sorted(random.sample(range(len(self.clayers[idx])), n_crossover_operations))
        child1_clayers = self.clayers[:]
        child2_clayers = parent2.clayers[:]

        for point in crossover_points:
            if point < len(child1_clayers[idx]) and point < len(child2_clayers[idx]):
                child1_clayers[idx][point], child2_clayers[idx][point] = (child2_clayers[idx][point],
                                                                          child1_clayers[idx][point])
            else:
                print(point, len(child1_clayers[idx]), len(child2_clayers[idx]))

        child1_clayers = Individual(child1_clayers, self.number_of_qubits)
        child2_clayers = Individual(child2_clayers, self.number_of_qubits)

        return child1_clayers, child2_clayers



