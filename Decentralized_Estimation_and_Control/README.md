# Implementation of Multi-Agent Coordination by Decentralized Estimation and Control
Citation: Yang, Peng, Randy A. Freeman, and Kevin M. Lynch. "Multi-agent coordination by decentralized estimation and control." IEEE Transactions on Automatic Control 53.11 (2008): 2480-2496.

## About
This repository is an implementation of the multi-agent distributed estimators and gradient controllers in the paper. Please refer to the paper for a detailed description of the algorithms.

## Getting Started
The main simulation loop runs in the `system_simpulator.py` file. The number of agents, the types of estimators, and the initial conditions for the simulation can be edited there.

The simulation uses `matplotlib` to render the simulation. The `data_plotter.py` file was taken and adapted from the Introduction to Controls course on GitHub (https://github.com/randybeard/controlbook_public).

## Known Limitations
I was never able to fully recreate the exact results of the paper. Some of the reasons might include:
- A bug in my code
- Incorrect implementation of the estimators or gradient controllers
- Differences in initial conditions (not defined in the paper)

Some places to start looking:
- The goal vector defined in the paper seems to be an invalid variance matrix, since it would have a 0 eigenvalue.
- The cost function, `J`, does go to zero, even though `f` does not exactly approach `f*` as time goes to infinity. This could mean that the cost function is being calculated wrong.