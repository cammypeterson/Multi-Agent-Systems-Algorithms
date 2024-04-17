import numpy as np
from scipy.integrate import ode
from Classes.multiagent_system import MultiAgentSystem
import os
from Classes.gif_generation import GIFGenerator
         

### SIMULATION PARAMETERS ###
t_start = 0
t_end = 1.2
Ts = 0.01
delta = 4.05
epsilon = 0.05
control_law = "rendezvous_simple" # "rendezvous_simple", "rendezvous_connected", or "formation_control"
generate_GIF = True

# define initial conditions (x, y corrdinates for each agent)
x0 = np.array([[-8., 0],
               [-6, 2],
               [-6, -2],
               [-4, 0],
               [0, 0],
               [4, 0],
               [6, 2],
               [6, -2],
               [8, 0]])

# instantiate GIFGenerator class and delete the files in the image/ directory
if generate_GIF:
    g = GIFGenerator()
    g.delete_frames()

# instantiate class object for the system
system = MultiAgentSystem(x0, delta, epsilon, control_law = control_law, create_gif = generate_GIF) 

# initialize ODE solver
r = ode(system.derivatives).set_integrator("dopri5")
r.set_initial_value(system.state.flatten(), t=t_start)

# loop through simulations
while r.successful():
    # animate the current state
    system.animate(r.t, Ts, xlim = (-10, 10), ylim = (-4, 4))
    
    # integrate to propogate the state one time step
    state = r.integrate(r.t + Ts)
    
    # update the adjacency matrix
    system.update_state_and_graph(state)
    
    # end condition
    if r.t >=  t_end:
        break


# Create the GIF
if generate_GIF:
    g.create_gif("GIF/" + control_law + ".gif")
    
input("Press Enter to exit")  
    