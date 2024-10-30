import os
import sys
from model_file import model
import time
import pickle
import tensorcircuit as tc
import toolbox as tlx
import genetic_algorithm as ga
from tools import save_individual, save_fitness, DAG_SELECT, express
from deap.tools.emo import sortNondominated as sort_nondominated
from constants import (btsize, NUMBER_OF_GENERATIONS, N_QUBITS, N_LAYERS,
                       POP_SIZE, n_classes, train_pop)


K = tc.set_backend("tensorflow")

n_qubits = N_QUBITS
n_layers = N_LAYERS
nbatch = btsize
number_of_generations = NUMBER_OF_GENERATIONS


def train_circuit(pop, g=None, indexs=None, n_classes=None):
    ind = pop[0]
    new_pop = []
    copy_pop = pop[1:]
    quantum_model = model.QuantumMLModel(ind)
    x_train, y_train, x_test, y_test = quantum_model.load_and_preprocess_data()
    quantum_model.train(x_train, y_train)
    loss, accuracy = quantum_model.evaluate(x_test, y_test)
    ind.fitness.values = [accuracy]
    ind.model_params = quantum_model.get_model_params()
    new_pop.append(ind)
    for individual in copy_pop:
        new_model = model.QuantumMLModel(individual)
        original_weights = quantum_model.get_model_params()
        new_model.set_model_params(original_weights)
        loss, accuracy = new_model.evaluate(x_test, y_test)

        individual.fitness.values = [accuracy]
        individual.model_params = quantum_model.get_model_params()
        new_pop.append(individual)

    Indexs = []
    Values = []

    for index, individual in enumerate(new_pop):
        Indexs.append(index)
        Values.append(individual.fitness.values[0])

    save_fitness(g, Indexs, Values, n_classes)
    save_individual(ind, g, indexs)
    return new_pop


def genetic_algorithm(pop, toolbox, number_of_generations, n_classes):
    new_pop = train_circuit(pop, g=0, indexs=0, n_classes=n_classes)
    pop = new_pop

    newpop = []
    for g in range(number_of_generations):
        print(f"Generation {g}/{number_of_generations}")

        non_dominated_solutions = sort_nondominated(pop, len(pop))[0]
        best_candidate = non_dominated_solutions[0]
        for ind in non_dominated_solutions:
            if best_candidate.fitness.values[0] > ind.fitness.values[0]:
                best_candidate = ind
        lucky_ind = [best_candidate]
        try:
            select_pop, evolve_pop = EVO.select_and_evolve(pop, toolbox)
            newpop.extend(evolve_pop)

            if len(newpop) + len(select_pop) < POP_SIZE:
                pop_size = POP_SIZE - (len(newpop) + len(select_pop))
                newpop.extend(EVO.new_pop(pop_size))

            dag_pop = DAG_SELECT(newpop, train_pop - len(select_pop))
            kl_pop = express(dag_pop)

            select_pop.extend(kl_pop)
            # select_pop.extend(newpop)
            select_pop.extend(lucky_ind)
            pop = select_pop

            new_pop = train_circuit(pop, g=g + 1, indexs=g+1, n_classes=n_classes)
            pop = new_pop

        except Exception as exc:
            print(f'Run generation {g}/{number_of_generations} encountered an exception: {exc}')

    return pop


print('n_classes', n_classes)
toolbox = tlx.initialize_toolbox()
EVO = ga.Evolution()
first_pop = EVO.new_pop()
dag_pop = DAG_SELECT(first_pop)
pop = express(dag_pop)
start = time.perf_counter()
try:
    pop = genetic_algorithm(pop, toolbox, number_of_generations, n_classes)
except ValueError as e:
    print(f"Exiting due to error: {e}")
    sys.exit(1)
