import numpy as np
from typing import Callable


def GetClosestNeighbors(i, agents, k):
    neighborDistances = []
    for j in range(len(agents)):
        if j == i:
            dist = np.inf
        else:
            dist = np.linalg.norm(agents[j].state[:2] - agents[i].state[:2])
        neighborDistances.append(dist)
    idx = np.argpartition(np.array(neighborDistances), k)
    neighbors = []
    for a in idx[:k]:
        neighbors.append(agents[a])
    return neighbors

class flock_dynamics:

    def Usubfunc(x):
        """Helper potential function so that the potential function is continuous, but not necessarily differentiable

        Args:
            x (_type_): dist between agents

        Returns:
            _type_: value of the potential function.
        """
        return 1/(x**2) + np.log(x**2)
    def U(x):
        """Potential function for graph with dynamic topology

        Args:
            x (_type_): usually is the distance between agents

        Returns:
            _type_: potential value
        """
        f = 0
        if x > 3:
            f = flock_dynamics.Usubfunc(3)
        else:
            f = flock_dynamics.Usubfunc(x)
        return f
    
    def Grad_U(x):
        """Derivative of Potential function, used in the control law

        Args:
            x (_type_): usually is the distance between agents

        Returns:
            _type_: potential derivative value
        """
        f = 0
        if x > 3:
            f = 0
        else:
            f = -2/x**3 + 2/x
        return -1/x + x

    def __init__(self, state: np.ndarray):
        self.state = state # x, y, xd, yd
        self.neighbors = []
        self.next_state = np.copy(state)
        self.history = [state]

    def RK4(f:Callable, x0:np.ndarray, u:np.ndarray, ts:float)->np.ndarray:
        """ Runga-Kutta 4 algorithm for integration

        Args:
            f (Callable): dynamics function we wish to integrate
            x0 (np.ndarray): starting value of the state that we wish to integrate
            u (np.ndarray): command for the system that we incorperate into dynamics
            ts (float): timestep for integration

        Returns:
            np.ndarray: updated state vector that integrates the time-derivative of the system
        """
        X1 = f(x0, u)
        X2 = f(x0 + ts/2 * X1, u)
        X3 = f(x0 + ts/2 * X2, u)
        X4 = f(x0 + ts * X3, u)
        return x0 + ts/6 * (X1 + 2*X2 + 2*X3 + X4)

    def cmd(self, state: np.ndarray) -> np.ndarray:
        """Generate the command based on the state of the flock

        Args:
            state (np.ndarray): state of the agent robot (x,y,xd,yd)

        Returns:
            command (np.ndarray): 
        """
        u = np.array([0., 0.])
        for n in self.neighbors:
            rj_mag = np.linalg.norm(state[:2] - n.state[:2])
            u += -(state[2:] - n.state[2:]) - flock_dynamics.Grad_U(rj_mag) * np.array([state[0] - n.state[0], state[1] - n.state[1]]) / rj_mag

        return u

    def dynamics(state: np.ndarray, u: np.ndarray)-> np.ndarray:
        """Double integrator dynamics.  Returns the xdot

        Args:
            state (np.ndarray): [x,y, xd, yd]
            u (np.ndarray): [xdd, ydd]

        Returns:
            np.ndarray: state derivative
        """
        state_d = np.copy(state)
        state_d[0:2] = state[2:]
        state_d[2:] = u
        return state_d

    def getHeading(state: np.ndarray) -> float:
        """Helper function for the heading

        Args:
            state (np.ndarray): agent state

        Returns:
            float: heading in radians
        """
        return np.arctan2(state[1], state[0])
    
    def setNextState(self, ts:float)->None:
        """Pre-loads the next state after RK4 integration.   Used so that all agents update simultaneously

        Args:
            ts (float): timestep
        """
        u = self.cmd(self.state)
        self.next_state = flock_dynamics.RK4(flock_dynamics.dynamics, self.state, u, ts)

    def update(self):
        """Make the state equal to the pre-loaded next state
        """
        self.state = self.next_state
        self.history.append(self.state)

    def getHistory(self):
        """Returns the history of the states

        Returns:
            _type_: _description_
        """
        return np.concatenate(self.history).reshape((-1,4))
    
    def getHeadingHistory(self):
        """Calculates the history of the heading from the history of the states

        Returns:
            _type_: _description_
        """
        hist = self.getHistory()
        thetaHist = np.arctan2(hist[:,3], hist[:,2])
        return thetaHist