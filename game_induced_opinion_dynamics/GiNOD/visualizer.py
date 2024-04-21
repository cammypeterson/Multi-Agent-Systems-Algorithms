"""
visualizer.py

Visualizer script for the results of the Game-induced Nonlinear
Opinion Dynamics (GiNOD) framework.
"""

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "./GiNOD/data/"

if __name__ == "__main__":

    # Visualization Parameters
    data_path = f"{DATA_DIR}/test_l0_t1"
    dt = 0.2
    
    road_bounds = (-1.5, 8.5)           # Road boundary limits
    toll_station_x_bounds = (50, 80)    # Toll station x-axis limits
    toll_station_2_py = 3.5             # Toll station 2 y-axis position
    toll_station_width = 3.5            # Toll station width

    # Load saved data
    xs = np.load(data_path + "_xs.npy")
    zs = np.load(data_path + "_zs.npy")
    Hs = np.load(data_path + "_Hs.npy")
    PoI = np.load(data_path + "_PoI.npy")

    n_steps = xs.shape[1]
    t = np.arange(0, n_steps*dt, dt)

    # Plot the results
    plt.figure()
    ax = plt.subplot(3,2,1)
    ax.plot(t, zs[0, :], label="$z_1^1$")
    ax.plot(t, zs[1, :], label="$z_2^1$")
    ax.set_ylabel("z")
    ax.set_ybound(-7, 7)
    ax.set_title("Player 1")
    ax.legend()

    ax = plt.subplot(3,2,2)
    ax.plot(t, zs[2, :], label="$z_1^2$")
    ax.plot(t, zs[3, :], label="$z_2^2$")
    ax.set_ybound(-7, 7)
    ax.set_title("Player 2")
    ax.legend()

    ax = plt.subplot(3,2,3)
    ax.plot(t, zs[4, :], label="$\lambda^1$")
    ax.set_ylabel("$\lambda$")
    ax.set_ybound(0, 5)
    ax.legend()
    
    ax = plt.subplot(3,2,4)
    ax.plot(t, zs[5, :], label="$\lambda^2$")
    ax.set_ybound(0, 5)
    ax.legend()

    ax = plt.subplot(3,2,5)
    ax.plot(t[:-1], PoI[0, :], label="$PoI^1$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PoI")
    ax.set_ybound(0, 5)
    ax.legend()
    
    ax = plt.subplot(3,2,6)
    ax.plot(t[:-1], PoI[1, :], label="$PoI^2$")
    ax.set_xlabel("Time (s)")
    ax.set_ybound(0, 5)
    ax.legend()

    plt.show()

    plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(xs[0, :], xs[1, :], label="Agent 1")
    ax.plot(xs[4, :], xs[5, :], label="Agent 2")
    ax.plot(xs[0,:], np.full_like(xs[0,:], road_bounds[0] - 1.5*toll_station_width), color='black')
    ax.plot(xs[0,:], np.full_like(xs[0,:], road_bounds[1] + 1.5*toll_station_width), color='black')
    ax.add_patch(plt.Rectangle((toll_station_x_bounds[0], toll_station_2_py - toll_station_width/2),
                                toll_station_x_bounds[1] - toll_station_x_bounds[0], toll_station_width,
                                color='gray', alpha=0.5))
    ax.add_patch(plt.Rectangle((toll_station_x_bounds[0], 3*toll_station_2_py - toll_station_width/2),
                                toll_station_x_bounds[1] - toll_station_x_bounds[0], toll_station_width,
                                color='gray', alpha=0.5))
    ax.add_patch(plt.Rectangle((toll_station_x_bounds[0], -1*toll_station_2_py - toll_station_width/2),
                                toll_station_x_bounds[1] - toll_station_x_bounds[0], toll_station_width,
                                color='gray', alpha=0.5))
    ax.legend()

    plt.show()

