from deap import creator, base, tools
# from genetic_net import evaluate_fitness
# from classification_tc.genetic_net.genetic_algorithm import Evolution, mutate_ind
from genetic_algorithm import Evolution, mutate_ind
from individual import Individual

# # 创建适应度类和个体类
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", Individual, fitness=creator.FitnessMax)

def initialize_toolbox():
    """初始化 DEAP 工具箱"""
    toolbox = base.Toolbox()

    # 注册个体和种群初始化方法
    toolbox.register("population", Evolution.new_pop)
    # 注册选择、交叉和变异操作方法
    toolbox.register("mate", Evolution.mate, toolbox=toolbox)
    toolbox.register("mutate_individuals", Evolution.mutate_individuals)
    toolbox.register("mutate_ind", mutate_ind)
    toolbox.register("select_and_evolve", Evolution.select_and_evolve)

    # 注册遗传函数
    toolbox.register("evolution", Evolution.evolution)
    # toolbox.register("evaluate", evaluate_fitness.main_evaluate)

    return toolbox