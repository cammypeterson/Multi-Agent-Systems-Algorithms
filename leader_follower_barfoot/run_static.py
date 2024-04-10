import numpy as np
import matplotlib.pyplot as plt
from trajectory import make_leader_trajectory, get_ref_traj, get_velocity, get_curvature, get_static, get_dynamic
from params import dt, ARROW_LENGTH

np.set_printoptions(suppress=True)

# Actual trajectory of leader (assume known)
xc, yc, thetac, vc, Kc, t, idx_so, idx_sf = make_leader_trajectory()

fig, ax = plt.subplots()
ax.plot(xc, yc, '-', color='red')
ax.plot(xc[0], yc[0], '*', color='red')
ax.plot(xc[-1], yc[-1], '*', color='red')
ax.plot(xc[idx_so], yc[idx_so], '*', color='red')
ax.plot(xc[idx_sf], yc[idx_sf], '*', color='red')

# Initial offsets
pi_all = [-2, 2, 2, -2]
qi_all = [2, -2, 2, -2]
num_agents = 4

for i in range(num_agents):

    pi = pi_all[i]
    qi = qi_all[i]

    # Shifted reference traj
    x_ref, y_ref, _, _ = get_ref_traj(xc, yc, thetac, vc, Kc, pi, idx_so, idx_sf, dt)
    vc_ref = get_velocity(x_ref, y_ref, dt)
    Kc_ref = get_curvature(x_ref, y_ref, dt)

    ax.plot(x_ref, y_ref, '-', color='blue')
    ax.plot(x_ref[0], y_ref[0], '*', color='blue')
    ax.plot(x_ref[-1], y_ref[-1], '*', color='blue')

    # Follower traj
    x, y, theta = get_static([xc[0]+pi, yc[0]+qi, thetac[0]], vc_ref, Kc_ref, qi, dt)

    ax.plot(x, y, '-', color='black', alpha=0.5)
    ax.plot(x[0], y[0], 'o', color='black')
    ax.plot(x[-1], y[-1], 'o', color='black')

plt.xlabel('x')
plt.ylabel('y')
plt.show()