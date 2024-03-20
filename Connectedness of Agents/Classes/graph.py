import numpy as np

# Class used to keep track of relevant adjacency matrices for rendezvous and formation control
class Graph():
    def __init__(self, x0, delta, epsilon):
        self.delta = delta # communication range of agents
        self.epsilon = epsilon # parameters (0 < epsilon < delta) for control laws
        self.n = len(x0) # number of agents
        self.dist_graph = np.zeros((self.n, self.n)) # keeps track of the distances between agents
        self.delta_graph = np.zeros((self.n, self.n)) # keeps track of which agents are within communication range of each other
        self.neighbors_graph = np.zeros((self.n, self.n)) # keeps track of which agents have been within (delta - epsilon) of each other at some point
        
        self.update(x0) # initialize the graphs
        
    # update all graphs
    def update(self, state):
        self.update_dist_graph(state) 
        self.update_delta_graph()
        self.update_neighbors_graph()
        
    # update dist_graph with current distances between agents
    def update_dist_graph(self, state):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.dist_graph[i, j] = np.linalg.norm(state[i] - state[j])
                self.dist_graph[j, i] = self.dist_graph[i, j]
    
    # update delta_graph to show agents that are within communication range of each other   
    def update_delta_graph(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.delta_graph[i, j] == 0:
                    if self.dist_graph[i, j] <= self.delta:
                        self.delta_graph[i, j] = 1
                        self.delta_graph[j, i] = 1
                else:
                    if self.dist_graph[i, j] > self.delta:
                        print("Graph disconnected!") # notify user if edge is loset
                        self.delta_graph[i, j] = 0
                        self.delta_graph[j, i] = 0
                
    # udpate neighbors_graph to show which agents have been within (delta - epsilon) at some point in the sim            
    def update_neighbors_graph(self):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.neighbors_graph[i, j] == 0:
                    if self.dist_graph[i, j] <= (self.delta - self.epsilon):
                        self.neighbors_graph[i, j] = 1
                        self.neighbors_graph[j, i] = 1
                else:
                    if self.delta_graph[i, j] == 0:
                        self.neighbors_graph[i, j] = 0
                        self.neighbors_graph[j, i] = 0
                
                    
