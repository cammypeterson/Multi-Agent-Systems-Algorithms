import numpy as np 
import matplotlib.pyplot as plt 

class Particle():
    
    def __init__(self, theta = 0, vel = 1, pos = 0 + 0j, w_0 = 0) -> None:
        self.theta = theta
        self.vel = vel
        self.pos = pos
        self.w_0 = w_0

        self.positions_real = []
        self.positions_imag = []

        self.circle_center = 0

        self.circle_centers_real = []
        self.circle_centers_imag = []

    def euclid_propogate(self, u):
        self.theta = self.theta + u
        self.pos = self.pos + np.cos(self.theta).item(0) + 1j*np.sin(self.theta).item(0)
        self.positions_real.append(self.pos.real)
        self.positions_imag.append(self.pos.imag)
        if self.w_0 == 0:
            self.circle_center = 0 + 0*1j
        else:
            self.circle_center = self.pos + 1j * 1/self.w_0 * np.exp(1j*self.theta)
        self.circle_centers_real.append(self.circle_center.real)
        self.circle_centers_imag.append(self.circle_center.imag)
        self.vel = np.exp(1j*self.theta)

if __name__ == "__main__":

    positions = []

    particle = Particle(pos = 4 + 1j)
   
    for i in range(50):
        positions.append(particle.pos)
        particle.euclid_propogate(.2)

    pos_real = [pos.real for pos in positions]
    pos_imag = [pos.imag for pos in positions]

    plt.plot(pos_real, pos_imag)
    plt.show()
