import numpy as np
import matplotlib.pyplot as plt
from dynamics1 import flock_dynamics, GetNeighbors

# Simulation Initialization
agents = []
numAgents = 5
k=3
t = 0.
t_end = 100.
ts = 0.05
times = [t]

# Initialize states
for i in range(numAgents):
    x0 = np.random.rand((4))-0.5
    # Random position
    x0[:2]*=20
    # Random Velocity
    x0[2:]*=3
    # Add to list
    agents.append(flock_dynamics(x0))

# simulate until t_end
while t < t_end:
    # Get the agents connected neighbors
    for i in range(len(agents)):
        agents[i].neighbors = GetNeighbors(i, agents, k)
    # Get the update ready for the whole graph
    for agent in agents:
        agent.setNextState(ts)
    # Update all agents simultaneously
    for agent in agents:
        agent.update()
    # Update Time
    t += ts
    times.append(t)

# Plotting Graphs
    
# Position Graph
plt.figure()
for agent in agents:
    hist = agent.getHistory()
    p = plt.plot(hist[:,0], hist[:,1])
    plt.arrow(hist[-1,0], hist[-1,1], hist[-1,2]/np.linalg.norm(hist[-1,2:]), hist[-1,3]/np.linalg.norm(hist[-1,2:]),shape='full', length_includes_head=True, head_width=0.2,color=p[-1].get_color())
    plt.grid(True)
    plt.title("Position")
    plt.xlabel("x")
    plt.ylabel("y")

# Heading Graph
plt.figure()
for agent in agents:
    hist = agent.getHeadingHistory()
    plt.plot(times, hist)
plt.grid(True)
plt.title("Heading")
plt.xlabel("t")
plt.ylabel(r"$\theta$")

# X difference plot
plt.figure()
for agent in agents:
    hist = agent.getHistory()
    for neighbor in agent.neighbors:
        nhist = neighbor.getHistory()
        plt.plot(times, hist[:,2] - nhist[:,2])
        
plt.grid(True)
plt.title("Difference between X Velocities")
plt.xlabel("t")
plt.ylabel("Velocity Difference")

# Y difference plot
plt.figure()
for agent in agents:
    hist = agent.getHistory()
    for neighbor in agent.neighbors:
        nhist = neighbor.getHistory()
        plt.plot(times, hist[:,3] - nhist[:,3])
plt.grid(True)
plt.title("Difference between Y Velocities")
plt.xlabel("t")
plt.ylabel("Velocity Difference")

plt.show()