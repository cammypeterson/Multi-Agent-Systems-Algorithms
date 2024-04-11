# This file contains the estimator and the controller
# for the system.

import numpy as np


class PIAgent:
    def __init__(self, agent_id, agents=None):
        '''
        Initiallizes an agent with the Non-Linear Gradient Controller with 
        PI estimator defined in Section III-C.

        @param: n = number of agents
        @param: agents -> array of all agents -> neighbors are filtered later.
        '''
        # Dimension of each state's space (i.e., p = [px, py] in this case)
        m = 2
        # dimension of the goal vector
        self.ell = m*(m+3) // 2
        # number of agents
        self.n = 0
        self.agents = agents
        self.id = agent_id
        # communication range
        self.comm_range = 15

        # Goal vector f*
        # Note that this is different than what the paper says, since the goal vector
        # defined in the paper is not a valid variance matrix
        self.f_star = np.array([[0., 0., 50., 50., 0.]]).T
        
        # Damping matrix B
        self.B = 100*np.eye(m)

        # Control gain matrix Gamma
        self.GAM = np.diag(np.array([80., 80., 8., 8., 8.]))

        # Damping gain matrix
        self.Lambda = np.zeros((self.ell, self.ell))
        # Scalar nonlinear damping gain
        self.c = 0.0

        # Estimator "forgetting factor"
        self.gamma = 6.
        # Estimator internal state
        self.aida = np.zeros((2*self.ell, 1))
        # estimator gains
        self.a0 = 20
        self.b0 = 0.2

        # State: xi = [pi.T, pi_dot.T].T
        self.state = np.array([[0., 0., 0., 0.]]).T

        # Simulation parameters
        self.Ts = 0.01

    def set_agents(self, agents):
        '''
        Initializes the agents. Must be called before the main simulation loop
        '''
        self.agents = agents
        self.n = len(agents)

    def set_initial_state(self, pos, vel):
        '''
        Initializes the state of the agent. Must be called before the main
        simulation loop, if nonzero
        '''
        self.state = np.array([[pos[0], pos[1], vel[0], vel[1]]]).T

    def get_phi(self):
        pi_ = self.state[0:2]
        return self.phi(pi_)
    
    def get_f_star(self):
        return self.f_star

    def uds(self, pi):
        '''
        Computes the upper-diagonal stack function as defined in the paper
        '''
        m = len(pi)
        M = pi @ pi.T
        result = np.array([])
        for i in range(m):
            result = np.append(result, np.diag(M,i))
        return np.reshape(result, (len(result), 1))
    
    def phi(self, pi):
        return np.vstack((pi, self.uds(pi))) 
    
    def F(self, state):
        '''
        Dynamics of the agent given by
        x_dot = f(xi, ui) = [pi_dot, ui]
        '''
        # Rename variables for clarity
        pi = state[0:2]
        pi_dot = state[2:]
        yi = self.R()

        # Calculate control input
        ui = -self.B @ pi_dot - self.Jphi(pi).T @ self.GAM @ (yi - self.f_star) \
            - self.Jphi(pi).T @ self.Lambda @ self.Jphi(pi) @ pi_dot - self.c * self.zeta(pi) * pi_dot

        x_dot = np.vstack((pi_dot, ui))
        return x_dot
    
    # The paper says this is a C^1 function that returns a Real number
    # I chose this C^1 function. Turns out it is not used in the paper's
    # simulation results, since self.c is zero
    def zeta(self, pi):
        return abs(pi[0][0])**2 + abs(pi[1][0])**2

    def Jphi(self, pi):
        '''
        Compute the Jacobian matrix of phi
        '''
        jacobian = np.array([[1, 0, 2*pi[0][0], 0, pi[1][0]],
                              [0, 1, 0, 2*pi[1][0], pi[0][0]]]).T
        return jacobian
    
    # Signal generator, G
    def G(self):
        pi_ = self.state[0:2]
        return np.vstack((pi_, self.aida))
    
    # Returns signal generator output
    def get_signal(self):
        return self.G()
    
    def Q(self, aida):
        '''
        Calculate aida_dot (the derivative of the internal estimator state)
        '''
        pi_ = self.state[0:2]
        vi_ = aida[:self.ell]
        wi_ = aida[self.ell:]
        sum1_ = 0
        sum2_ = 0
        sum3_ = 0
        for j in range(len(self.agents)):
            if j != self.id:
                # Get message from agent
                message = self.agents[j].get_signal()
                pj_ = message[:2]
                vj_ = message[2:2+self.ell]
                wj_ = message[2+self.ell:]

                # Calculate the gain variables a, b
                a = self.calculate_a(pi_, pj_)
                b = self.calculate_b(pi_, pj_)
                sum1_ += a * (vi_ - vj_)
                sum2_ += b * (wi_ - wj_)
                sum3_ += b * (vi_ - vj_)

        vi_dot_ = -self.gamma * vi_ - sum1_ + sum2_ + self.gamma * self.phi(pi_)
        wi_dot_ = -sum3_
        aida_dot_ = np.vstack((vi_dot_, wi_dot_))
        return aida_dot_

    def calculate_a(self, pi, pj):
        if np.linalg.norm(pi - pj) <= self.comm_range:
            return self.a0
        return 0.
    
    def calculate_b(self, pi, pj):
        if np.linalg.norm(pi - pj) <= self.comm_range:
            return self.b0
        return 0.
    
    def R(self):
        v_ = self.aida[:self.ell]
        return v_

    def update(self):
        '''
        Main update loop. Integrates the estimated global state
        of the system and then the dynamics of the agent using
        an RK4 algorithm.
        '''
        if self.agents == None:
            raise ValueError('Agents not set!')
        else:
            # update aida
            self.rk4_step_aida()
            # update xdot (propogate dynamics)
            self.rk4_step_x()
            return self.state

    # Integrate ODE using Runge-Kutta RK4 algorithm
    def rk4_step_aida(self):
        Q1 = self.Q(self.aida)
        Q2 = self.Q(self.aida + self.Ts / 2 * Q1)
        Q3 = self.Q(self.aida + self.Ts / 2 * Q2)
        Q4 = self.Q(self.aida + self.Ts * Q3)
        self.aida += self.Ts / 6 * (Q1 + 2 * Q2 + 2 * Q3 + Q4)
        
    # Integrate ODE using Runge-Kutta RK4 algorithm
    def rk4_step_x(self):
        F1 = self.F(self.state)
        F2 = self.F(self.state + self.Ts / 2 * F1)
        F3 = self.F(self.state + self.Ts / 2 * F2)
        F4 = self.F(self.state + self.Ts * F3)
        self.state += self.Ts / 6 * (F1 + 2 * F2 + 2 * F3 + F4)


if __name__=='__main__':
    # controller = Agent(7, 2, )
    pass