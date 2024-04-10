import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, odeint

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from trajectory import make_leader_trajectory, get_ref_traj, get_velocity, get_curvature, get_static, get_dynamic
from params import dt 

# Eq. 15
def get_q(s, so, sf, qo, qf):

    q = []
    dq = qf - qo
    ds = sf - so
    b = (s - so)/ds

    # Get q velocity (Eq. 16)
    for i in range(len(s)):

        s_val = s[i]
        b_val = b[i]

        if s_val >= 0 and s_val <= so:
            q.append(qo)
        elif s_val > so and s_val <= sf:
            q.append(qo + dq*b_val**2*(3 - 2*b_val))
        else:
            q.append(qf)

    return np.array(q)

# Eq. 16
def get_dqds(s, so, sf, qo, qf):
    dq_ds = []
    dq = qf - qo
    ds = sf - so
    b = (s - so)/ds

    for i in range(len(s)):

        s_val = s[i]
        b_val = b[i]

        if s_val > so and s_val <= sf:
            dq_ds.append(6*(dq/ds)*b_val*(1-b_val))
        else:
            dq_ds.append(0)

    return np.array(dq_ds)

# Eq. 17
def get_d2qds2(s, so, sf, qo, qf):
    d2q_ds2 = []
    dq = qf - qo
    ds = sf - so
    b = (s - so)/ds

    for i in range(len(s)):

        s_val = s[i]
        b_val = b[i]

        if s_val > so and s_val <= sf:
            d2q_ds2.append(6*(dq/(ds**2))*(1-2*b_val))
        else:
            d2q_ds2.append(0)

    return np.array(d2q_ds2)


def get_distance(v, d0, dt):
    d = np.zeros(len(v))
    d[0] = d0
    for i in range(1, len(v)):
        d[i] = d[i - 1] + v[i - 1] * dt

    return d


# Get leader info
xc, yc, thetac, vc, Kc, t, idx_so, idx_sf = make_leader_trajectory()

fig, ax = plt.subplots()
ax.plot(xc, yc, '-', color='red')
ax.plot(xc[0], yc[0], '*', color='red')
ax.plot(xc[-1], yc[-1], '*', color='red')

# Example dynamic maneuver: Increase q over time (small square to big square)
pi_all = [2, 2, -2, -2]
qio_all = [0.5, -0.5, 0.5, -0.5]    # original distance from C
qif_all = [2.0, -2.0, 2.0, -2.0]    # final distance from C
numagents = 4

# Dynamic formations
for i in range(numagents):

    pi = pi_all[i]
    qi = qio_all[i]

    # Shifted reference traj
    x_ref, y_ref, idx_so_ref, idx_sf_ref = get_ref_traj(xc, yc, thetac, vc, Kc, pi, idx_so, idx_sf, dt)
    vc_ref = get_velocity(x_ref, y_ref, dt)
    Kc_ref = get_curvature(x_ref, y_ref, dt)

    ax.plot(x_ref, y_ref, '-', color='blue')
    ax.plot(x_ref[0], y_ref[0], '*', color='blue')
    ax.plot(x_ref[-1], y_ref[-1], '*', color='blue')

    # Dynamic params
    qo = qio_all[i]
    qf = qif_all[i]

    # Get new vc, Kc based on offset
    d0 = d0 = np.sqrt(x_ref[0]**2 + y_ref[0]**2)
    s = get_distance(vc_ref, d0, dt) 
    so = s[idx_so_ref]
    sf = s[idx_sf_ref] 

    # q info
    q = get_q(s, so, sf, qo, qf)
    dq_ds = get_dqds(s, so, sf, qo, qf)
    d2q_ds2 = get_d2qds2(s, so, sf, qo, qf)

    # Offsets
    x, y, theta = get_dynamic([xc[0] + pi, yc[0] + qi, thetac[0]], vc_ref, Kc_ref, q, dq_ds, d2q_ds2, dt)
    ax.plot(x, y, '-', color='black')
    ax.plot(x[0], y[0], 'o', color='black')
    ax.plot(x[-1], y[-1], 'o', color='black')

    ax.plot(x[idx_so_ref], y[idx_so_ref], 'o', color='blue')
    ax.plot(x[idx_sf_ref], y[idx_sf_ref], 'o', color='blue')

ax.set_xlabel('x')
ax.set_ylabel('x')
plt.show()