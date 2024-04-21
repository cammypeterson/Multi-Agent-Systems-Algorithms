# Consensus-Based Decentralized Auctions for Robust Task Allocation
Implementation based on the paper by Han-Lim Choi, Luc Brunet, and Jonathan P. How.

H. -L. Choi, L. Brunet and J. P. How, "Consensus-Based Decentralized Auctions for Robust Task Allocation," in IEEE Transactions on Robotics, vol. 25, no. 4, pp. 912-926, Aug. 2009, doi: 10.1109/TRO.2009.2022423

## Running Instructions
Installations of ```numpy``` and ```matplotlib``` are the only requirements to run this code.

Custom task allocation scenarios can be configured by changing the appropriate variables in the ```CBBA_main.py``` file. Examples of changeable parameters are number of agents, number of tasks, maximum number of tasks per agent, etc.

## Notes
If changing the number of agents, note that the default communication graph is hard coded for a five agent system in the ```CBBA_main.py``` file, so a custom communication graph for the desired number of agents would be needed. A completely untested (but not very complicated) ```graph_gen()``` function exists that should generate a random connected graph for arbitrary numbers of agents, so if you want to play around with that it should work with some (hopefully) minor debugging.

The simulation is set to run for a predefined number of iterations, but will likely reach convergence before hitting that limit. Convergence can be observed when the allocated tasks/paths for all the agents remain unchanged over consecutive iterations.

The reward function used by default in ```CBBA.py``` is a time-based reward, awarding more reward for completing tasks faster. A custom reward function to be used for task allocation may be substituted by writing a new ```score_path()``` function.
