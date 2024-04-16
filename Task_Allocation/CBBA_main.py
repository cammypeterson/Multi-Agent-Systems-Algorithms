import numpy as np
from CBBA import CBBA_Network, Agent
import matplotlib.pyplot as plt

def gen_graph():
    graph = np.zeros((Nu, Nu))
    graph_done = False
    edge_gen_prob = 0.2
    while not graph_done:
        for i in range(Nu):
            for j in range(Nu - i):
                if i != j:
                    if np.random.uniform() > edge_gen_prob:
                        graph[i, i + j] = 1
                        graph[i + j, i] = 1
        check_irr = np.sum([np.linalg.matrix_power(graph, i) for i in range(graph.shape[0])])
        if np.all(check_irr):
            if np.random.uniform() > 0.5:
                graph_done = True

    return graph

def plot_cbba(CBBA, W, step):
    tasks = CBBA.tasks
    plt.figure()
    plt.xlim(0, W)
    plt.ylim(0, W)
    plt.title(f"Step {step}")
    plt.scatter(tasks[:, 0], tasks[:, 1], c='b', s=100)
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, CBBA.Nu)))
    for i in range(len(CBBA.agents)):
        color = next(colors)
        agent = CBBA.agents[i]
        plt.scatter(agent.pos[0], agent.pos[1], color=color, s=100)
        if len(agent.p) > 0:
            tasks_path = np.array([tasks[task, :] for task in agent.p])
            tasks_path = np.vstack((agent.pos, tasks_path))
            plt.plot(tasks_path[:, 0], tasks_path[:, 1], color=color)
    plt.show()



if __name__ == "__main__":
    Nt = 10  # num tasks
    Nu = 5  # num agents
    Lt = 5  # max num of tasks per agent
    W = 2e3  # 2 km wide area
    dim = 2  # 2D space, should convert easily to 3D
    vel = 40  # m/s, constant velocity of all agents

    tasks = np.random.uniform(size=(Nt, dim)) * W
    # tasks = np.array([[77.24022765, 114.36328034],  # Fixed task positions for debugging
    #                  [132.95957243, 1425.24326958],
    #                  [1460.83095506, 236.10149446],
    #                  [499.11768873, 1613.42453588],
    #                  [166.20213617, 370.93351231]])
    # tasks = np.array([[1744.10409331, 131.40242204],  # 10 fixed tasks for debugging
    #                  [1894.51279127, 407.03820918],
    #                  [783.55961519, 856.56562503],
    #                  [69.06109692, 282.39447721],
    #                  [540.69543275, 210.51253841],
    #                  [1150.68335554, 1488.96016374],
    #                  [795.60594883, 160.56357511],
    #                  [1740.12065043, 1769.26626549],
    #                  [445.58348537, 1529.9887469],
    #                  [172.07796695, 411.73850113]])

    agents = []
    # agent_pos_s = np.array([[1080.41857925, 1890.75481568],  # Fixed agent positions for debugging
    #                         [160.82824788, 676.23214351],
    #                         [193.16516382, 323.52414919],
    #                         [755.5806528,  1513.92576691],
    #                         [1899.48203228, 1506.06647737]])
    # agent_pos_s = np.array([[885.12648882, 400.91286275],  # Fixed agent positions for 10 task debugging
    #                         [1003.56015797, 789.46473292],
    #                         [47.68074273, 1006.97954067],
    #                         [680.87807643, 1898.14665203],
    #                         [840.35562494, 914.55085982]])

    for i in range(Nu):
        agent_pos = np.random.uniform(size=dim) * W
        # agent_pos = agent_pos_s[i, :]
        agent = Agent(Nt, Nu, Lt, tasks, i, agent_pos, vel)
        agents.append(agent)
        # print(agent_pos)

    graph = np.array([[0, 1, 0, 1, 0],  # static set graph for testing
                      [1, 0, 1, 1, 0],
                      [0, 1, 0, 0, 1],
                      [1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0]])
    # graph = np.ones((Nu, Nu))  # fully connected graph

    # graph = gen_graph()  # random communication graph -- NOT TESTED

    CBBA = CBBA_Network(Nt, Nu, Lt, agents, tasks, graph)

    num_steps = 20

    plot_cbba(CBBA, W, 0)
    for i in range(num_steps):
        CBBA.step()

        print(f"--------------------------Step {i+1}-----------------------------")
        for agent in CBBA.agents:
            print(f"Agent {agent.agent_num} - path: {agent.p}")
            print(f"bundle: {agent.b}")
        plot_cbba(CBBA, W, i+1)

