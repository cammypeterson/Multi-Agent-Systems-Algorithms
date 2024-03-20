from Classes.graph import Graph
import matplotlib.pyplot as plt
import numpy as np

# Class designed to simulate different control laws that maintain graph connectedness
class MultiAgentSystem():
    def __init__(self, x0, delta, epsilon, control_law = "rendezvous_connected", G_desired = None, x_desired = None, create_gif = True):
        self.graph = Graph(x0, delta, epsilon) # this graph object keeps track of multiple agency matrices relevant to the simulation (see Graph object for more info)
        self.delta = delta # parameter that defines the distance within which two agents are considered connected (AKA communication range)
        self.epsilon = epsilon # parameter used by control law to keep graph connected ( 0 < epsilon < delta)
        self.state = x0 # current state of the sim
        self.n, self.dim = x0.shape # number of agents and dimensionality of the state of each agent
        self.control_law = control_law # which control law will be used ("rendezvous_simple", "rendezvous_connected", or "formation_control")
        
        # setup to generate GIF, if desired
        self.create_gif = create_gif
        if self.create_gif:
            self.frame_itr = 0 # used to name file for saving each frame of animation
                    
        # state machine setup
        if self.control_law == "formation_control":
            self.phase = "rendezvous" # start with rendezvous
            
            # check to see that all parameters for formation control are given
            if G_desired is None or x_desired is None:
                raise ValueError("For formation control, a G_desired graph and x_desired positions must be given")
            
            self.graph.G_desired = G_desired # the desired formation connections (in adjacency matrix form)
            self.x_desired = x_desired # the desired target positions for each agent
    
    # generates a frame/image representing the current state of the sim
    def animate(self, t, Ts, xlim, ylim):
        plt.clf() # clear figure
        plt.scatter(self.state[:, 0], self.state[:, 1]) # plot agents
        
        # plot agent connections
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.graph.neighbors_graph[i, j]:
                    plt.plot([self.state[i, 0], self.state[j, 0]], [self.state[i, 1], self.state[j, 1]], color = 'r')
        
        plt.figtext(0.8, 0.71, f"t = {t:.2f}s")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        # same images to folder for GIF genearation
        if self.create_gif:
            if self.frame_itr < 10:
                plt.savefig(f"images/frame_00{self.frame_itr}.png")
            elif self.frame_itr < 100:
                plt.savefig(f"images/frame_0{self.frame_itr}.png")
            else:
                plt.savefig(f"images/frame_{self.frame_itr}.png")
            self.frame_itr += 1
            
        plt.pause(Ts) # run the GUI event loop for Ts seconds
                    
    # re-shapes the state array and updates the graph object
    def update_state_and_graph(self, state):
        self.state = state.reshape((self.n, self.dim))
        self.graph.update(self.state)
    
    # Derivative of the edge-tension function. This defines the influence agent_j has on the control law to ensure
    # that the communication graph of all the agents remains connected 
    def dVij(self, delta, l_ij_norm):
        return (2*delta - l_ij_norm)/(delta - l_ij_norm)**2

    # calculates the derivatives of the state (called by ODE solver to integrate and propogate the state)
    def derivatives(self, t, state):
        
        # check switching condition for formation_control
        if self.control_law == 'formation_control' and self.phase == 'rendezvous':
            for i in range(self.n):
                num_neighbors = sum(self.graph.neighbors_graph[i, :])
                max_dist = np.max(self.graph.dist_graph[i, :])
                if num_neighbors == self.n - 1 and max_dist <= (self.delta - self.epsilon)/2:
                    self.phase = "formation"
                    break
        
        state = state.reshape(int(len(state)/self.dim), self.dim)
        u = np.zeros(np.shape(state))
        for i in range(self.n):
            for j in range(self.n):
                
                # rendezvous control law
                if self.control_law == 'rendezvous_connected' or (self.control_law == 'formation_control' and self.phase == 'rendezvous'):
                    if self.graph.neighbors_graph[i, j]:
                        l_ij_norm = np.linalg.norm(state[i] - state[j])
                        u[i] -= self.dVij(self.delta, l_ij_norm)*(state[i] - state[j])
                
                # (simple) rendezvous control law (does not guarantee connectedness)
                elif self.control_law == 'rendezvous_simple':
                    if self.graph.delta_graph[i, j]:
                        u[i] -= (state[i] - state[j])
                        
                # formation control law
                elif self.control_law == 'formation_control' and self.phase == 'formation':
                    if self.graph.G_desired[i, j]:
                        d_ij = self.x_desired[i] - self.x_desired[j]
                        d_ij_norm = np.linalg.norm(d_ij)
                        l_d_norm = np.linalg.norm((state[i] - state[j] - d_ij))
                        u[i] -= self.dVij(self.delta - d_ij_norm, l_d_norm)*(state[i] - state[j] - d_ij)

        return u.flatten() # return derivatives of the states (which are equal to)