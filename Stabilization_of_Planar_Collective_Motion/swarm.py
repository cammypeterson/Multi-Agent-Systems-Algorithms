from typing import List
import numpy as np 
import matplotlib.pyplot as plt 
from particle import Particle
import random

class Swarm():
    
    def __init__(self, particles : List[Particle], w_0 : float) -> None:
        self.particles = particles
        self.N = len(self.particles)
        self.control = np.zeros((self.N, 1))
        self.w_0 = w_0
        self.del_U = np.zeros((self.N, 1))

        self.R = 0

    def compute_control_grad(self, K : float) -> None:
        # Equation 8

        for index, particle in enumerate(self.particles):
            self.control[index] = self.w_0 - K/self.N * sum([np.sin(-part.theta + particle.theta).item(0) for part in self.particles])

    def compute_control_phase_symmetry(self, kappa, K):
        # Equation 19
        
        self.find_center_of_mass()
        
        for index, particle in enumerate(self.particles):

            del_U_1 = 1/self.N*sum([np.sin(-part.theta + particle.theta).item(0) for part in self.particles])

            del_U = -K*del_U_1

            self.control[index] = self.w_0*(1+kappa*((np.exp(1j*particle.theta).real*(particle.pos - self.R).real)+(np.exp(1j*particle.theta).imag*(particle.pos - self.R).imag))) + kappa*del_U_1 + del_U
    
    def parrallel_to_circular(self, kappa, K, R_0):
        # Equation 63 (19 with beacon R_0)
        
        for index, particle in enumerate(self.particles):

            del_U_1 = 1/self.N*sum([np.sin(-part.theta + particle.theta).item(0) for part in self.particles])

            del_U = -K*del_U_1

            self.control[index] = self.w_0*(1+kappa*((np.exp(1j*particle.theta).real*(particle.pos - R_0).real)+(np.exp(1j*particle.theta).imag*(particle.pos - R_0).imag))) + del_U
    
    def circular_to_circular(self, kappa, K, R_0):
        # Equation 63 (19 with beacon R_0)
        
        for index, particle in enumerate(self.particles):

            del_U_1 = 1/self.N*sum([np.sin(-part.theta + particle.theta).item(0) for part in self.particles])

            del_U = -K*del_U_1

            self.control[index] = self.w_0*(1+kappa*((np.exp(1j*particle.theta).real*(particle.pos - R_0).real)+(np.exp(1j*particle.theta).imag*(particle.pos - R_0).imag))) + del_U
    
    def circular_to_parrallel(self, theta_0, w_0, kappa, d, particle, K):
        # Eq (66)

        # K < 0, d > 0

        self.find_center_of_mass()

        r_tilde_N = particle.pos - self.R

        del_U_1 = 1/self.N*sum([np.sin(part.theta - particle.theta).item(0) for part in self.particles])

        theta_N = particle.theta

        inner_product = r_tilde_N.real * particle.vel.real + r_tilde_N.imag * particle.vel.imag 

        u_N = w_0*(1+kappa*inner_product) + kappa*del_U_1 - K,del_U_1 + d*np.sin(theta_0 - theta_N)

        return u_N
    
    def compute_control_relative_equilibria(self, kappa, K_m):
        # Equation 37
        
        self.find_center_of_mass()
        
        for index, particle in enumerate(self.particles):

            del_U = self.calc_del_U(K_m, swarm.N, index) 

            del_U_1 = 1/self.N*sum([np.sin(part.theta - particle.theta).item(0) for part in self.particles])

            self.control[index] = self.w_0*(1+kappa*((np.exp(1j*particle.theta).real*(particle.pos - self.R).real)+(np.exp(1j*particle.theta).imag*(particle.pos - self.R).imag))) - kappa*del_U_1 + del_U

    def calc_del_U(self, K_m_0, M, k):

        del_U = 0.
        K_m = K_m_0
        
        for j in range(self.N):

            del_U_m = []

            for m in range(1,M+1):

                if m > swarm.N/2:
                    K_m = 0

                del_U_m.append(K_m/float(m) * np.sin(m*(swarm.particles[k].theta-swarm.particles[j].theta)))

            del_U += sum(del_U_m)
            K_m = K_m_0

        return del_U

    def find_center_of_mass(self):

        self.R = 0

        for particle in self.particles:
            self.R += particle.pos

        self.R /= len(self.particles)

    def euclid_propogate(self):

        for control, particle in zip(self.control, self.particles):
            particle.euclid_propogate(control)

if __name__ == "__main__":

    K = -0.05 # 0.0125/4.0 # Use this for the splay state.
    w_0 = 1/25.0 # 0.1
    kappa = 0.1

    particles = []

    for _ in range(12):
        particles.append(Particle(random.uniform(0,2*np.pi), 1, complex(random.uniform(0,10), random.uniform(0,10)), w_0))

    swarm = Swarm(particles, w_0)

    num_steps = 200 # must be larger than 100

    swarm.w_0 = w_0
    
    for t in range(num_steps):

        # swarm.compute_control_grad(K)
        swarm.compute_control_phase_symmetry(kappa, K)
        # swarm.compute_control_relative_equilibria(kappa, K)
        # swarm.circular_to_parrallel(90*np.pi/180.0, w_0, kappa, d, swarm.particles[-1],K) # K < 0, d >0, w_0 = kappa = 0

        swarm.euclid_propogate()

    theta_0 = 45*np.pi/180.0

    # IMPULSE

    for particle in swarm.particles:
        particle.theta = theta_0
        particle.vel = np.exp(1j*particle.theta)

    # Circular-to-parrallel
    
    K = -0.1 # 0.0125/4.0 # Use this for the splay state.
    w_0 = 0
    kappa = 0
    d = 0.1
    swarm.w_0 = w_0

    for t in range(num_steps):

        # swarm.compute_control_grad(K)
        swarm.compute_control_phase_symmetry(kappa, K)
        # swarm.compute_control_relative_equilibria(kappa, K)
        swarm.circular_to_parrallel(theta_0, w_0, kappa, d, swarm.particles[-1],K) # K < 0, d > 0, w_0 = kappa = 0

        swarm.euclid_propogate()
    
    theta_0 = (15-90)*np.pi/180.0

    # IMPULSE

    for particle in swarm.particles:
        particle.theta = theta_0
        particle.vel = np.exp(1j*particle.theta)

    # Parrallel-to-parrallel

    K = -0.1 # 0.0125/4.0 # Use this for the splay state.
    w_0 = 0
    kappa = 0
    d = 0.1
    swarm.w_0 = w_0

    for t in range(num_steps):

        # swarm.compute_control_grad(K)
        swarm.compute_control_phase_symmetry(kappa, K)
        # swarm.compute_control_relative_equilibria(kappa, K)
        swarm.circular_to_parrallel(theta_0, w_0, kappa, d, swarm.particles[-1],K) # K < 0, d > 0, w_0 = kappa = 0

        swarm.euclid_propogate()

    theta_0 = 45*np.pi/180.0

    # IMPULSE

    for particle in swarm.particles:
        particle.theta = theta_0
        particle.vel = np.exp(1j*particle.theta)

    # Parrallel-to-parrallel

    K = -0.1 
    w_0 = 0
    kappa = 0
    d = 0.1
    swarm.w_0 = w_0

    for t in range(num_steps):

        # swarm.compute_control_grad(K)
        swarm.compute_control_phase_symmetry(kappa, K)
        # swarm.compute_control_relative_equilibria(kappa, K)
        swarm.circular_to_parrallel(theta_0, w_0, kappa, d, swarm.particles[-1],K) # K < 0, d > 0, w_0 = kappa = 0

        swarm.euclid_propogate()

    # Parrallel-to-circular

    K = -0.1 
    w_0 = 1/25.0
    kappa = .1
    d = 0.1
    swarm.w_0 = w_0

    # IMPULSE
    swarm.find_center_of_mass()

    R_0 = swarm.R

    for particle in swarm.particles:
        num = 1j*w_0*(particle.pos - R_0) # Find better explanation and physical meaning.
        complex_arg = np.arctan2(num.imag,num.real)
        particle.theta = complex_arg - particle.theta
        particle.vel = np.exp(1j*particle.theta)


    for t in range(num_steps):

        # swarm.compute_control_grad(K)
        # swarm.compute_control_phase_symmetry(kappa, K)
        # swarm.compute_control_relative_equilibria(kappa, K)
        # swarm.circular_to_parrallel(theta_2, w_0, kappa, d, swarm.particles[-1],K) # K < 0, d > 0, w_0 = kappa = 0
        swarm.parrallel_to_circular(kappa,K,R_0)

        swarm.euclid_propogate()

    theta_0 = np.pi
    
    # Circular-to-circular

    K = -0.0125/4.0 # Use this for the splay state.
    w_0 = 1/50.0
    kappa = .2
    swarm.w_0 = w_0
    swarm.find_center_of_mass()

    R_0 = swarm.R

    for t in range(3*num_steps):

        # swarm.compute_control_grad(K)
        swarm.compute_control_phase_symmetry(kappa, K)
        # swarm.compute_control_relative_equilibria(kappa, K)
        # swarm.circular_to_parrallel(theta_0, w_0, kappa, d, swarm.particles[-1],K) # K < 0, d > 0, w_0 = kappa = 0
        swarm.circular_to_circular(kappa,K,R_0)

        swarm.euclid_propogate()



    arrow_length = 10
    head_width = 1
    head_length = 2

    for particle in swarm.particles:
        plt.plot(particle.positions_real, particle.positions_imag)
        plt.scatter(particle.positions_real[-1], particle.positions_imag[-1])
        # plt.scatter(particle.circle_centers_real[-1], particle.circle_centers_imag[-1]) # Plot circle centers.
        plt.arrow(particle.positions_real[-1], particle.positions_imag[-1], arrow_length*np.cos(particle.theta).item(0),
                  arrow_length*np.sin(particle.theta).item(0), head_width=head_width, head_length=head_length, fc='black', ec='black')

    plt.axis('equal')
    plt.title('Swarm Dynamics')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()

