# DiNNO
Custom Pytorch Lightning Implementation of the Distributed Neural Network Optimization for Multi-Robot Collaborative Learning algorithm ([Link to videos and Github](https://msl.stanford.edu/projects/dist_nn_train)).

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10.6 installed
- A virtual environment manager (e.g., `venv`, `conda`)

## Installation

To install the necessary dependencies and set up the environment, follow these steps:

1. Create and activate a virtual environment:

   For `venv`:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   For `conda`:
   ```
   conda create --name myenv python=3.10.6
   conda activate myenv
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the project, navigate to the project directory and execute the `run.py` file with the necessary arguments. For example:

```
python run.py --config=configs/mnist_config.yaml
```

Replace `configs/mnist_config.yaml` with the path to with a custom configuration file if different.

