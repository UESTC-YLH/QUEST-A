import copy

import numpy.random
import random
from deap import creator, base

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)


class Individual1:
    """This class is a container for an individual in GA."""

    def __init__(self, cnots, args, ESL=2.0):
        self.args = args
        self.number_of_qubits = 5
        self.permutation = random.sample(range(self.number_of_qubits), self.number_of_qubits)
        self.ESL = ESL
        self.cnots = cnots
        self.model_params = None

    def __len__(self):
        return len(self.cnots)

    def generate_random_cnots(self, initialize=False):
        """
        生成一个随机的CNOT门组合，其中每个门的control和target位置都在N_QUBITS个量子比特中随机选择。
        """
        if initialize:
            p = 1 / 30
        else:
            p = 1 / self.ESL
        n_cnot = numpy.random.geometric(p)
        combinations = []
        for _ in range(n_cnot):
            control = random.randrange(self.number_of_qubits)
            target = random.randrange(self.number_of_qubits)
            # 确保control和target不相同
            while target == control:
                target = random.randrange(self.number_of_qubits)
            combinations.append((control, target))
        return combinations

    def generate_cnot(self):
        control = random.randrange(self.number_of_qubits)
        target = random.randrange(self.number_of_qubits)
        # 确保control和target不相同
        while target == control:
            target = random.randrange(self.number_of_qubits)
        return (control, target)

    def discrete_uniform_mutation(self):
        """
        随机选择一个门，并随机替换为另一组门。
        """
        if not self.cnots:
            return
        idx = random.randrange(len(self.cnots))
        self.cnots[idx] = self.generate_random_cnots()

    def continuous_uniform_mutation(self):
        """
        随机选择一个门，并随机修改其control和target位置。
        """
        if not self.cnots:
            return
        idx = random.randrange(len(self.cnots))
        control = random.randrange(self.number_of_qubits)
        target = random.randrange(self.number_of_qubits)
        # 确保control和target不相同
        while target == control:
            target = random.randrange(self.number_of_qubits)
        self.cnots[idx] = (control, target)

    def sequence_insertion(self):
        """
        在随机位置插入一个新的门。
        """
        new_cnot = self.generate_cnot()
        idx = random.randrange(len(self.cnots) + 1)
        # 检查新门是否已经存在于列表中，如果存在则不插入
        if new_cnot not in self.cnots:
            self.cnots.insert(idx, new_cnot)

    def sequence_and_inverse_insertion(self):
        """
        在随机位置插入一个新的门以及其逆操作。
        """
        new_cnot = self.generate_cnot()
        idx = random.randrange(len(self.cnots) + 1)
        # 检查新门是否已经存在于列表中，如果存在则不插入
        if new_cnot not in self.cnots:
            self.cnots.insert(idx, new_cnot)
            self.cnots.insert(idx + 1, (new_cnot[1], new_cnot[0]))

    def insert_mutate_invert(self):
        """
        在随机位置插入一个新的门，或者变异一个门，或者对一个门进行逆操作。
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
        随机删除一个门。
        """
        if not self.cnots:
            return
        idx = random.randrange(len(self.cnots))
        del self.cnots[idx]

    def sequence_replacement(self):
        """
        随机选择一个门，并替换为一组新的门。
        """
        if not self.cnots:
            return
        idx = random.randrange(len(self.cnots))
        self.cnots[idx] = self.generate_random_cnots()

    def sequence_swap(self):
        """
        随机选择两个门，并交换它们的位置。
        """
        if len(self.cnots) < 2:
            return
        idx1, idx2 = random.sample(range(len(self.cnots)), 2)
        self.cnots[idx1], self.cnots[idx2] = self.cnots[idx2], self.cnots[idx1]

    def sequence_scramble(self):
        """
        随机选择一部分门，并随机交换它们的排列顺序。
        """
        if len(self.cnots) < 2:
            return
        random.shuffle(self.cnots)

    def permutation_mutation(self):
        """
        对门的顺序进行变异，相当于重新排列门的顺序。
        """
        random.shuffle(self.cnots)

    def move_gate(self):
        """
        移动一个门到另一个位置。
        """
        if not self.cnots:
            return
        idx = random.randrange(len(self.cnots))
        new_idx = random.randrange(len(self.cnots))
        self.cnots.insert(new_idx, self.cnots.pop(idx))

    def cross_over1(self, parent2, toolbox):
        """This function gets two parent solutions, creates an empty child, randomly
        picks the number of gates to be selected from each parent and selects that
        number of gates from the first parent, and discards that many from the
        second parent. Repeats this until parent solutions are exhausted.
        """
        self_circuit = self.cnots
        parent2_circuit = parent2.cnots
        p1 = p2 = 1.0

        if len(self_circuit) != 0:
            p1 = self.args.EMC / len(self.cnots)
        if (p1 <= 0) or (p1 > 1):
            p1 = 1.0

        if len(parent2_circuit) != 0:
            p2 = parent2.EMC / len(parent2.clayers)
        if (p2 <= 0) or (p2 > 1):
            p2 = 1.0

        clayers = []
        child = creator.Individual(clayers, self.number_of_qubits)
        child.clayers = []
        turn = 1
        while len(self_circuit) or len(parent2_circuit):
            if turn == 1:
                number_of_gates_to_select = numpy.random.geometric(p1)
                child.clayers += self_circuit[:number_of_gates_to_select]
                turn = 2
            else:
                number_of_gates_to_select = numpy.random.geometric(p2)
                child.clayers += parent2_circuit[:number_of_gates_to_select]
                turn = 1
            self_circuit = self_circuit[number_of_gates_to_select:]
            parent2_circuit = parent2_circuit[number_of_gates_to_select:]
        return child  # creator class


def cross_over(parent1, parent2, args):
    """
    两个父辈的交叉函数。
    """
    # 交叉点，随机选择
    crossover_point = random.randint(1, min(len(parent1.cnots), len(parent2.cnots)))

    # 初始化子代
    child1_cnots = []
    child2_cnots = []

    # 为子代添加门，同时保证门的唯一性
    for idx in range(len(parent1.cnots)):
        # 检查是否到达交叉点
        if idx < crossover_point:
            # 将父辈的门添加到子代中
            if parent1.cnots[idx] not in child1_cnots:
                child1_cnots.append(parent1.cnots[idx])
            if parent2.cnots[idx] not in child2_cnots:
                child2_cnots.append(parent2.cnots[idx])
        else:
            # 将父辈2的门添加到子代1中
            if parent2.cnots[idx] not in child1_cnots:
                child1_cnots.append(parent2.cnots[idx])
            # 将父辈1的门添加到子代2中
            if parent1.cnots[idx] not in child2_cnots:
                child2_cnots.append(parent1.cnots[idx])

    # 创建子代对象
    child1 = Individual(child1_cnots, args.n)
    child2 = Individual(child2_cnots, args.n)

    return child1, child2


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
        """
        生成一个随机的CNOT门组合，其中每个门的control和target位置都在N_QUBITS个量子比特中随机选择。
        """
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
        """
        移动一个clayer到另一个位置。
        """
        if not self.clayers:
            return
        idx = random.randrange(len(self.clayers))
        new_idx = random.randrange(len(self.clayers))
        self.clayers.insert(new_idx, self.clayers.pop(idx))

    def cross_over(self, parent2):
        """
        两个父辈的交叉函数。
        """
        # 随机选择交叉操作类型：0表示在层与层之间交叉，1表示在层内进行 cnot 交叉
        crossover_type = random.randint(0, 1)

        if crossover_type == 0:
            # 在层与层之间进行交叉操作
            return self.crossover_layers(parent2)
        else:
            # 在层内进行 cnot 交叉操作
            return self.crossover_within_layers(parent2)

    def crossover_layers(self, parent2):
        """
        在层与层之间进行交叉操作。
        """
        # 确定交叉点的数量
        # print('self.clayers', self.clayers) # <deap.creator.Individual object at 0x75a0067082b0>
        n_crossover_points = min(len(self.clayers), len(parent2.clayers))

        # 随机选择一个交叉点
        crossover_point = random.randint(0, n_crossover_points - 1)

        # 初始化子代
        child1_clayers = self.clayers[:crossover_point] + parent2.clayers[crossover_point:]
        child2_clayers = parent2.clayers[:crossover_point] + self.clayers[crossover_point:]

        # 创建子代对象
        child1_clayers = Individual(child1_clayers, self.number_of_qubits)
        child2_clayers = Individual(child2_clayers, self.number_of_qubits)

        return child1_clayers, child2_clayers

        # 随机选择一个交叉点
        # crossover_point = random.randint(0, N_LAYERS-1)
        #
        # A = copy.copy(self.clayers.pop(crossover_point))
        # B = copy.copy(parent2.clayers.pop(crossover_point))
        # del self.clayers[crossover_point]
        # del parent2.clayers[crossover_point]
        # print('AB', A, B)  # [(0, 3), (1, 3), (2, 3)
        # # self.clayers[crossover_point] = B
        # # parent2.clayers[crossover_point] = A
        # # self.clayers.insert(crossover_point, self.clayers.pop(idx))
        # child1_clayers = self.clayers.insert(crossover_point, B)
        # child2_clayers = parent2.clayers.insert(crossover_point, A)
        #
        # # 创建子代对象
        # child1 = Individual(child1_clayers)
        # child2 = Individual(child2_clayers)
        #
        # return child1, child2

    def crossover_within_layers(self, parent2):
        """
        在层内进行 cnot 交叉操作。
        """
        # 随机选择一个层进行交叉操作
        idx = random.randint(0, min(len(self.clayers), len(parent2.clayers)) - 1)

        # 获取交叉操作的数量
        # n_crossover_operations = numpy.random.geometric(1 / self.ESL)
        n_crossover_operations = 2

        # 确定交叉点的位置
        crossover_points = sorted(random.sample(range(len(self.clayers[idx])), n_crossover_operations))

        # 初始化子代
        child1_clayers = self.clayers[:]
        child2_clayers = parent2.clayers[:]

        # 进行交叉操作
        for point in crossover_points:
            # print('[idx][point]', crossover_points, [idx], [point])  # [idx][point] [0, 1, 4] [3] [0]
            if point < len(child1_clayers[idx]) and point < len(child2_clayers[idx]):
                child1_clayers[idx][point], child2_clayers[idx][point] = (child2_clayers[idx][point],
                                                                          child1_clayers[idx][point])
            else:
                print(point, len(child1_clayers[idx]), len(child2_clayers[idx]))


        # 创建子代对象
        child1_clayers = Individual(child1_clayers, self.number_of_qubits)
        child2_clayers = Individual(child2_clayers, self.number_of_qubits)

        return child1_clayers, child2_clayers

        # # 随机选择一个层进行交叉操作
        # idx = random.randint(0, N_LAYERS - 1)
        #
        # # 获取交叉操作的数量
        # n_crossover_operations = numpy.random.geometric(1 / self.ESL)
        #
        # # 确定交叉点的位置
        # crossover_points = sorted(random.sample(range(len(self.clayers[idx])), n_crossover_operations))
        #
        # # 初始化子代
        # child1_clayers = self.clayers[:]
        # child2_clayers = parent2.clayers[:]
        #
        # # 进行交叉操作
        # for point in crossover_points:
        #     child1_clayers[idx][point], child2_clayers[idx][point] = child2_clayers[idx][point], child1_clayers[idx][point]
        #
        # # 创建子代对象
        # child1 = Individual(child1_clayers)
        # child2 = Individual(child2_clayers)
        #
        # return child1, child2
