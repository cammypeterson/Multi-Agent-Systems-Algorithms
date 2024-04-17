# Game-induced Nonlinear Opinion Dynamics

---
### Introduction

This repository contains code to implement the algorithm described in the paper "Emergent Coordination through Game-Induced Nonlinear Opinion Dynamics"

The preprint for this paper can be found at [arxiv.org](https://arxiv.org/abs/2304.02687)
The original version of the code by the authors can also be accessed in the authors' [original repository](https://github.com/SafeRoboticsLab/opinion_game)

---
### Installation

To install the required packages:

1. Clone the repository
2. create or activate a conda or virtual environment
3. Go to the top level directory of the repository and run the command:

    ```bash
    pip install . -e
    ```

---

### Usage

To run the code, simply activate your python environment and run the `sim.py` file.
This will run the simulations with the parameters specified inside the `sim.py` file, and output the results to a `data` folder.

The `visualizer.py` file can be used to visualize the results of the simulations as saved to the `data` folder.
