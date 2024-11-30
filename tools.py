import os
import csv
import time
import pickle
import networkx as nx
import numpy as np
# from deap import tools
import concurrent.futures
import tensorcircuit as tc
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import genetic_net.toolbox as tlx
# import classification_tc.genetic_net.genetic_algorithm as ga
import torchvision.transforms as transforms
# from concurrent.futures import ProcessPoolExecutor
import torch
import torchvision
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import entropy
from constants import (N_QUBITS, N_LAYERS, POP_SIZE, train_pop,
                       n_classes, num_dag, num_KL)
from projectq.ops import (H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, CX, Rx,
                          Ry, Rz, SqrtX, get_inverse, Swap, SwapGate)
import chemistry_VQE.genetic_net.toolbox as tlx
import chemistry_VQE.genetic_net.genetic_algorithm as ga


toolbox = tlx.initialize_toolbox()
EVO = ga.Evolution()

def DAG_SELECT(pop, trainpop=None):
    t1 = time.perf_counter()
    if trainpop is None:
        Train_pop = num_dag
    else:
        Train_pop = trainpop
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # 使用executor.map来并行处理所有个体
    #     futures = {executor.submit(process_individual, ind): ind for ind in pop}
    #     results = []
    #     for future in concurrent.futures.as_completed(futures):
    #         results.append(future.result())
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_individual, ind): ind for ind in pop}
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x[1])

    # 计算中位数附近的索引范围
    n = len(results)
    if n % 2 == 0:  # 如果总数是偶数
        mid_index = n // 2 - 1  # 取中间两个数的左边一个
    else:
        mid_index = n // 2  # 如果总数是奇数

    # 确定选取的起始和结束索引
    start_index = max(0, mid_index - 10)  # 确保起始索引不小于0
    end_index = min(n, mid_index + 11)  # 确保结束索引不大于列表长度

    # 选取中位数附近的train_pop个个体ind
    new_pop = [result[0] for result in results[start_index:end_index] if end_index - start_index >= Train_pop]

    # 确保new_pop的长度是train_pop
    if len(new_pop) > Train_pop:
        new_pop = new_pop[:Train_pop]  # 如果选取的个体超过train_pop个，取前train_pop个
    elif len(new_pop) < Train_pop:
        # 如果选取的个体不足train_pop个，需要调整start_index和end_index来补足
        while len(new_pop) < Train_pop and start_index > 0:
            start_index -= 1
            new_pop.insert(0, results[start_index][0])
        while len(new_pop) < Train_pop and end_index < n:
            end_index += 1
            new_pop.append(results[end_index - 1][0])

    # 将 new_pop 按照路径数进行排序，大的在前小的在后
    new_pop.sort(key=lambda x: next(result[1] for result in results if result[0] == x), reverse=True)
    print('get DAG_SELECT time:', time.perf_counter() - t1)
    return new_pop


def process_individual(ind):
    quantum_circuit = []
    produced_circuits = ind.clayers
    for j in range(N_LAYERS):
        produced_circuit = produced_circuits[j]
        for op in produced_circuit:
            if op[0] == "TFG":
                if op[1] in [CX, CNOT]:
                    quantum_circuit.append(('CX', [op[2], op[3]]))
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

            elif op[0] == "SFG":
                # can be H,X,Y,Z,T,T^d,S,S^d,sqrtX,sqrtXdagger
                if op[1] == H:
                    quantum_circuit.append(('H', [op[2]]))
                elif op[1] == X:
                    quantum_circuit.append(('x', [op[2]]))
                elif op[1] == Y:
                    quantum_circuit.append(('y', [op[2]]))
                elif op[1] == Z:
                    quantum_circuit.append(('z', [op[2]]))
                elif op[1] == T:
                    quantum_circuit.append(('t', [op[2]]))
                elif op[1] == Tdagger:
                    quantum_circuit.append(('tdg', [op[2]]))
                elif op[1] == S:
                    quantum_circuit.append(('s', [op[2]]))
                elif op[1] == Sdagger:
                    quantum_circuit.append(('sdg', [op[2]]))
                elif op[1] == SqrtX:
                    quantum_circuit.append(('sx', [op[2]]))
                elif op[1] == get_inverse(SqrtX):
                    quantum_circuit.append(('sxdg', [op[2]]))
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

            elif op[0] == "SG":
                # can be Rx,Ry,Rz
                if op[1] == Rx:
                    quantum_circuit.append(('rx', [op[2]]))
                elif op[1] == Ry:
                    quantum_circuit.append(('ry', [op[2]]))
                elif op[1] == Rz:
                    quantum_circuit.append(('rz', [op[2]]))
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

    # 创建有向图
    dag = nx.DiGraph()
    last_op = {}
    initial_nodes = set()

    # 添加节点和边到有向图
    for idx, (gate, qubits) in enumerate(quantum_circuit):
        node = f"{gate}_{idx}"
        dag.add_node(node)
        for qubit in qubits:
            if qubit not in last_op:
                initial_nodes.add(node)
            else:
                dag.add_edge(last_op[qubit], node)
            last_op[qubit] = node

    # 所有最晚操作的节点作为终点
    end_nodes = set(last_op.values())

    # 计算所有起点到终点的路径数，并记录路径
    all_paths = []
    for start_node in initial_nodes:
        for end_node in end_nodes:
            paths = list(nx.all_simple_paths(dag, source=start_node, target=end_node))
            all_paths.extend(paths)

    number_of_paths = len(all_paths)
    print('number_of_paths', number_of_paths)
    # 返回计算结果
    return ind, number_of_paths


def express(pop, num_samples=500):
    t1 = time.perf_counter()
    expressibility_values = []

    for ind in pop:
        sampled_states = []
        for _ in range(num_samples):
            params = generate_random_params(N_LAYERS, N_QUBITS)
            circ = sample_quantum_states(ind.clayers, params)
            state = circ.wavefunction()
            sampled_states.append(state)

        fidelities = compute_fidelities(sampled_states)
        expressibility_value = compute_expressibility(fidelities)
        expressibility_values.append((ind, expressibility_value))

    print('expressibility_values', expressibility_values)
    # 排序并筛选出表达性最好的前num_KL个个体
    expressibility_values.sort(key=lambda x: abs(x[1]), reverse=False)
    top_individuals = [ind for ind, _ in expressibility_values[:num_KL]]
    print('get express time:', time.perf_counter() - t1)
    return top_individuals

# def express(pop, num_samples=500):
#     expressibility_values = []
#
#     def compute_expressibility_for_ind(ind):
#         sampled_states = []
#         for _ in range(num_samples):
#             params = generate_random_params(N_LAYERS, N_QUBITS)
#             circ = sample_quantum_states(ind.clayers, params)
#             state = circ.wavefunction()
#             sampled_states.append(state)
#
#         fidelities = compute_fidelities(sampled_states)
#         expressibility_value = compute_expressibility(fidelities)
#         return ind, expressibility_value
#
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(compute_expressibility_for_ind, pop))
#
#     expressibility_values.extend(results)
#
#     # print('expressibility_values', expressibility_values)
#     # 排序并筛选出表达性接近于零的前num_KL个个体
#     expressibility_values.sort(key=lambda x: abs(x[1]), reverse=False)
#     top_individuals = [ind for ind, _ in expressibility_values[:num_KL]]
#
#     return top_individuals


def sample_quantum_states(clayers, params):
    c = tc.Circuit(N_QUBITS)
    for j in range(N_LAYERS):
        for i in range(N_QUBITS):
            c.rz(i, theta=params[j, i, 0])
            c.ry(i, theta=params[j, i, 1])
            c.rz(i, theta=params[j, i, 2])

        produced_circuit = clayers[j]
        for op in produced_circuit:
            if op[0] == "TFG":
                if op[1] in [CX, CNOT]:
                    c.cx(op[2], op[3])
                elif op[1] in [Swap, SwapGate]:
                    c.swap(op[2], op[3])
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

            elif op[0] == "SFG":
                pass

            elif op[0] == "SG":
                pass

    return c


def generate_random_params(n_layers, n_qubits):
    return np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))


def compute_fidelities(states):
    fidelities = []
    num_states = len(states)
    for i in range(num_states):
        for j in range(i + 1, num_states):
            fid = fidelity(states[i], states[j])
            fidelities.append(fid)
    return fidelities


def fidelity(state1, state2):
    return np.abs(np.vdot(state1, state2)) ** 2


def haar_fidelity_distribution(F, N):
    return (N - 1) * (1 - F) ** (N - 2)


def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def compute_expressibility(fidelities):
    hist, bin_edges = np.histogram(fidelities, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    haar_dist = haar_fidelity_distribution(bin_centers, 2 ** N_QUBITS)

    return np.abs(-kl_divergence(hist, haar_dist))


# 确保moving_average函数返回的是numpy数组
def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size) / window_size, mode='valid')


# 构建保真度分布
def construct_fidelity_distribution(clayers, params_list):
    fidelities = []
    num_samples = len(params_list)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            state1 = sample_quantum_states(clayers, params_list[i])
            state2 = sample_quantum_states(clayers, params_list[j])
            fid = fidelity(state1, state2)
            fidelities.append(fid)
    return np.histogram(fidelities, bins=50, range=(0, 1), density=True)[0]


# 定义 Haar 随机态分布
def haar_random_state_distribution(n_bins, n_dim):
    bins = np.linspace(0, 1, n_bins + 1)
    haar_dist = [(n_dim - 1) * (1 - F) ** (n_dim - 2) for F in bins]
    haar_dist = np.array(haar_dist) / np.sum(haar_dist)  # 归一化
    return haar_dist


# 计算 KL 散度
def compute_kl_divergence(clayers, params_list):
    P_C_F = construct_fidelity_distribution(clayers, params_list)
    P_Haar_F = haar_random_state_distribution(len(P_C_F), 2 ** N_QUBITS)
    KL_div = entropy(P_C_F, P_Haar_F)
    return -KL_div  # 返回负的 KL 散度作为可表达性


def save_individual(ind, generation, index):
    clayers_path = fr'/home/cpu_user_cpu/ylh_temporary/mnist/{N_LAYERS}_layers/{n_classes}_class//model/{generation}_{index}_clayers.pkl'
    model_params_path = fr'/home/cpu_user_cpu/ylh_temporary/mnist/{N_LAYERS}_layers/{n_classes}_class//model/{generation}_{index}_model_params.pth'
    # clayers_path = fr'/mnt/shared/CPUData/ylh_cpu/mnist/{N_LAYERS}_layers/{n_classes}_class//model/{generation}_{index}_clayers.pkl'
    # model_params_path = fr'/mnt/shared/CPUData/ylh_cpu/mnist/{N_LAYERS}_layers/{n_classes}_class//model/{generation}_{index}_model_params.pth'

    os.makedirs(os.path.dirname(clayers_path), exist_ok=True)
    with open(clayers_path, 'wb') as clayers_file:
        pickle.dump(ind.clayers, clayers_file)
    torch.save(ind.model_params, model_params_path)


def load_individual(generation, index):
    clayers_path = fr'/home/cpu_user_cpu/ylh_temporary/mnist/{N_LAYERS}_layers/{n_classes}_class//model//{generation}_{index}_clayers.pkl'
    model_params_path = fr'/home/cpu_user_cpu/ylh_temporary/mnist/{N_LAYERS}_layers/{n_classes}_class//model//{generation}_{index}_model_params.pth'
    # clayers_path = fr'/mnt/shared/CPUData/ylh_cpu/mnist/{N_LAYERS}_layers/{n_classes}_class//model//{generation}_{index}_clayers.pkl'
    # model_params_path = fr'/mnt/shared/CPUData/ylh_cpu/mnist/{N_LAYERS}_layers/{n_classes}_class//model//{generation}_{index}_model_params.pth'

    # 读取clayers内容
    with open(clayers_path, 'rb') as clayers_file:
        clayers = pickle.load(clayers_file)

    model_params = torch.load(model_params_path)
    return clayers, model_params


def save_fitness(generation, index, value, n_classes):
    file_path = fr"/home/cpu_user_cpu/ylh_temporary/output/mnist/{N_LAYERS}_layers/{n_classes}_class//fitness/{n_classes}n_classes{generation}.csv"
    # file_path = fr"/mnt/shared/CPUData/ylh_cpu/mnist/{N_LAYERS}_layers/{n_classes}_class//fitness/{n_classes}n_classes{generation}.csv"

    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Check if the file is empty and write the header if needed
        if csvfile.tell() == 0:
            csv_writer.writerow(["Index", "Value"])

        for i, v in zip(index, value):
            csv_writer.writerow([i, v])


def load_fitness(generation, idx, n_classes):
    file_path = fr"/home/cpu_user_cpu/ylh_temporary/mnist/5_layers/2_class//fitness/{n_classes}n_classes_{generation}.csv"
    # file_path = fr"/mnt/shared/CPUData/ylh_cpu/ouut/mnist/5_layers/2_class//fitness/{n_classes}n_classes_{generation}.csv"

    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row and row[0].isdigit() and int(row[0]) == idx:
                return row[1] if len(row) > 1 else None

    return None


def plot_losses_accuracies(losses, accuracies, acc, g=None, index=None):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    filename = f'/home/cpu_user_cpu/ylh_temporary/mnist/{N_LAYERS}_layers/{n_classes}_class//{acc}_{g}_{index}.png' if g is not None and index is not None else 'mnist/default_filename.png'
    # filename = f'/mnt/shared/CPUData/ylh_cpu/mnist/{N_LAYERS}_layers/{n_classes}_class//{acc}_{g}_{index}.png' if g is not None and index is not None else 'output/default_filename.png'
    plt.savefig(filename)


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Convert data to numpy format
    x_train, y_train = trainset.data.numpy(), trainset.targets.numpy()
    x_test, y_test = testset.data.numpy(), testset.targets.numpy()

    # Set labels to filter
    array = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    selected_labels = random.sample(array, n_classes)

    def filter_labels(x, y, labels):
        keep = np.isin(y, labels)
        x, y = x[keep], y[keep]

        label_map = {label: idx for idx, label in enumerate(labels)}
        y = np.array([label_map[label] for label in y])

        return x, y

    x_train, y_train = filter_labels(x_train, y_train, selected_labels)
    x_test, y_test = filter_labels(x_test, y_test, selected_labels)

    # Reshape data to (N, 28*28) and scale
    x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Standardize the data before PCA
    scaler = StandardScaler()
    x_train_flat = scaler.fit_transform(x_train_flat)
    x_test_flat = scaler.transform(x_test_flat)

    # Apply PCA to reduce dimensionality to 10
    pca = PCA(n_components=10)
    x_train_pca = pca.fit_transform(x_train_flat)
    x_test_pca = pca.transform(x_test_flat)

    # Convert numpy arrays to PyTorch tensors
    x_train_torch = torch.tensor(x_train_pca, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    x_test_torch = torch.tensor(x_test_pca, dtype=torch.float32)[:1000]
    y_test_torch = torch.tensor(y_test, dtype=torch.long)[:1000]

    # # Verify shapes
    # print(x_train_torch.shape)  # Should be (12000, 10)
    # print(x_test_torch.shape)  # Should be (1000, 10)
    # print(y_train_torch.shape)  # Should be ([12000])
    # print(y_test_torch.shape)  # Should be (1000,)
    return x_train_torch, y_train_torch, x_test_torch, y_test_torch

def genetic_algorithm(pop, toolbox, number_of_generations, n_classes, train_circuit, molecule_name=None):
    """
    :param pop: 初始种群
    :param toolbox: 定义遗传算法操作的工具箱，包括选择、交叉和变异等操作
    :param number_of_generations: 遗传算法的迭代代数，即进行选择、交叉和变异操作的次数
    :return: 最终种群
    """
    new_pop = train_circuit(pop, generation=0, indexs=0)
    pop = new_pop

    newpop = []
    print('start generating')
    for g in range(number_of_generations):
        print(f"Generation {g}/{number_of_generations}: len pop:{len(pop)}")
        select_pop, evolve_pop = EVO.select_and_evolve(pop, toolbox)

        if len(newpop) + len(select_pop) < POP_SIZE:
            pop_size = POP_SIZE - (len(newpop) + len(select_pop))
            newpop.extend(EVO.new_pop(pop_size))

        dag_pop = DAG_SELECT(newpop, train_pop - len(select_pop))
        kl_pop = express(dag_pop)

        select_pop.extend(kl_pop)

        new_pop = train_circuit(select_pop, generation=g + 1, indexs=g + 1)
        pop = new_pop

    return pop


