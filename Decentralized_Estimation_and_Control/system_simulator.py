import numpy as np
import matplotlib.pyplot as plt
from high_pass_agent import HighPassAgent
from pi_estimator_agent import PIAgent
from high_pass_kinematic_agent import HighPassKinematicAgent
from data_plotter import dataPlotter


class Simulator:
    def __init__(self, agents, is_remove_agent=False):
        '''
        Initializes the agents and objects needed for simulator
        '''
        self.agents = agents
        self.ti = 0.
        self.tf = 100
        self.Ts = 0.1
        self.colors = ['b','g','r','c','y','m','k']
        self.plotter = dataPlotter(len(agents))
        self.is_remove_agent = is_remove_agent

    def simulate(self):
        '''
        Main loop of the simulator. Iterates through each agent's dynamics at each 
        time step, until the final simulation time
        '''
        t = self.ti
        state_vect = []
        for i in range(len(self.agents)):
            state_vect.append([])
        
        while t < self.tf:
            f = np.zeros((5,1))
            for i in range(len(self.agents)):
                state_vect[i] = agents[i].update()
                f += agents[i].get_phi()
            f /= len(self.agents)
            
            # Update the data plotter
            f_star = self.agents[0].get_f_star()
            self.plotter.update(t, f_star, f, state_vect)

            if t > 25.0 and self.is_remove_agent:
                self.agents.pop(-1)
                # for i in range(len(self.agents)):
                #     self.agents[i].set_agents(self.agents)
                self.is_remove_agent = False
            t += self.Ts
            plt.pause(0.001)
        
        plt.waitforbuttonpress()
        plt.close()


if __name__=='__main__':
    # Construct n agents
    n = 7

    # Type of agent to simulate. Uncomment the line for the desired agent type
    # agents = [HighPassAgent(i) for i in range(n)]
    agents = [PIAgent(i) for i in range(n)]
    # agents = [HighPassKinematicAgent(i) for i in range(n)]

    for i in range(n):
        agents[i].set_agents(agents)
        # Arrange agents in a circular pattern
        pos_x = 10.*np.cos(2*(i+1)*np.pi/(len(agents)+0.5))
        pos_y = 10.*np.sin(2*(i+1)*np.pi/(len(agents)+0.5))

        # Arrange agents randomly
        # pos_x = np.sqrt(50)*(2*np.random.rand() - 1.)
        # pos_y = np.sqrt(50)*(2*np.random.rand() - 1.)

        vel_x = 0.
        vel_y = 0.
        agents[i].set_initial_state([pos_x, pos_y], [vel_x, vel_y])

    sim = Simulator(agents, is_remove_agent=True)
    sim.simulate()