import numpy as np
from params import *

def make_leader_trajectory():

    # Initialize position and orientation
    x = 0.0
    y = 0.0
    theta = 0.0

    # Lists to store trajectory
    x_traj = [x]
    y_traj = [y]
    v_traj = [v_straight]
    K_traj = [0]
    theta_traj = [theta]
    t = [0]
    time = 0

    # Straight phase
    for _ in range(int(straight_duration1 / dt)):
        x += v_straight * np.cos(theta) * dt
        y += v_straight * np.sin(theta) * dt
        x_traj.append(x)
        y_traj.append(y)
        theta_traj.append(theta)
        v_traj.append(v_straight)
        K_traj.append(0)
        time+=dt
        t.append(time)

    idx_so = len(x_traj)

    # Turning phase
    for _ in range(int(turn_duration1 / dt)):
        x += v_turn * np.cos(theta) * dt
        y += v_turn * np.sin(theta) * dt
        theta += omega1 * dt
        x_traj.append(x)
        y_traj.append(y)
        theta_traj.append(theta)
        v_traj.append(v_turn)
        K_traj.append(curvature1)
        time+=dt
        t.append(time)

    idx_sf = len(x_traj)
    
    # Straight phase
    for _ in range(int(straight_duration2 / dt)):
        x += v_straight * np.cos(theta) * dt
        y += v_straight * np.sin(theta) * dt
        x_traj.append(x)
        y_traj.append(y)
        v_traj.append(v_straight)
        K_traj.append(0)
        theta_traj.append(theta)
        time+=dt
        t.append(time)

    # Turning phase
    for _ in range(int(turn_duration2 / dt)):
        x += v_turn * np.cos(theta) * dt
        y += v_turn * np.sin(theta) * dt
        theta += omega2 * dt
        x_traj.append(x)
        y_traj.append(y)
        theta_traj.append(theta)
        v_traj.append(v_turn)
        K_traj.append(curvature2)
        time+=dt
        t.append(time)

    # Straight phase
    for _ in range(int(straight_duration3 / dt)):
        x += v_straight * np.cos(theta) * dt
        y += v_straight * np.sin(theta) * dt
        x_traj.append(x)
        y_traj.append(y)
        v_traj.append(v_straight)
        K_traj.append(0)
        theta_traj.append(theta)
        time+=dt
        t.append(time)

    v_traj = np.array(v_traj)
    K_traj = np.array(K_traj)

    return x_traj, y_traj, theta_traj, v_traj, K_traj, t, idx_so, idx_sf

def get_velocity(x, y, dt):

    xdot = np.diff(x) / dt
    ydot = np.diff(y) / dt

    v = np.sqrt(xdot**2 + ydot**2)

    return v

def get_curvature(x, y, dt):

    # Velocity
    dx_dt = np.diff(x) / dt
    dy_dt = np.diff(y) / dt

    # Acceleration
    d2x_dt2 = np.diff(dx_dt)  / dt
    d2y_dt2 = np.diff(dy_dt)  / dt

    # Make same length
    dx_dt = dx_dt[0:-1]
    dy_dt = dy_dt[0:-1]
    
    numerator = dx_dt * d2y_dt2 - d2x_dt2 * dy_dt
    denominator = (dx_dt**2 + dy_dt**2)**(3/2)

    K =  np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    return K

def get_pos(pos0, v, K, dt):

    # Get heading (Eq. 7)
    thetadoti = v*K
    thetai = np.zeros(len(thetadoti))
    thetai[0] = pos0[2]
    for i in range(1, len(thetadoti)):
        thetai[i] = thetai[i - 1] + thetadoti[i - 1] * dt

    # Get follower velocities
    xdoti = v*np.cos(thetai)
    ydoti = v*np.sin(thetai)

    # Get follower positions
    xi = np.zeros(len(thetadoti))
    yi = np.zeros(len(thetadoti))

    xi[0] = pos0[0]
    yi[0] = pos0[1]

    # Integrate velocities to get positions
    for i in range(1, len(thetadoti)):
        xi[i] = xi[i - 1] + xdoti[i - 1] * dt
        yi[i] = yi[i - 1] + ydoti[i - 1] * dt
        thetai[i] = thetai[i - 1] + thetadoti[i - 1] * dt

    return xi, yi, thetai

# si = dc + pi (Eq. 9)
def get_ref_traj(xc, yc, thetac, vc, Kc, p, idx_so, idx_sf, dt):

    numsteps_offset = round(abs(p /( vc[0] *dt))) 

    if p < 0:
        vc1 = np.ones_like(vc[0:numsteps_offset])*vc[0]
        Kc1 = np.ones_like(Kc[0:numsteps_offset])*Kc[0]

        # Offset beginning by extending
        x1, y1, theta1 = get_pos([xc[0]+p,yc[0], thetac[0]], vc1, Kc1, dt)

        # Offset end by truncating
        x2, y2, theta2 = get_pos([x1[-1],y1[-1], theta1[-1]], vc[0:-numsteps_offset], Kc[0:-numsteps_offset], dt)
        
        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2)) 

        idx_so_new = idx_so + numsteps_offset
        idx_sf_new = idx_sf + numsteps_offset

    elif p > 0:

        # Offset beginning by truncating
        x1, y1, theta1 = get_pos([xc[0]+p,yc[0], thetac[0]], vc[numsteps_offset:-1], Kc[numsteps_offset:-1], dt)

        # Offset end by extending
        vc2 = np.ones_like(vc[0:numsteps_offset])*vc[-1]
        Kc2 = np.ones_like(Kc[0:numsteps_offset])*Kc[-1]

        x2, y2, theta2 = get_pos([x1[-1],y1[-1], theta1[-1]], vc2, Kc2, dt)  

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2)) 

        idx_so_new = idx_so - numsteps_offset
        idx_sf_new = idx_sf - numsteps_offset

    else:
        x = xc
        y = yc

        idx_so_new = idx_so
        idx_sf_new = idx_sf 

    return x, y, idx_so_new, idx_sf_new

def get_static(pos0, vc, Kc, qi, dt):

    # May be different lens due to differentiation methods
    trunc_idx = min(len(vc), len(Kc))
    vc = vc[0:trunc_idx]
    Kc = Kc[0:trunc_idx]

    # Get controls (Eq. 6)
    vi = vc * (1-qi*Kc)
    Ki = Kc / (1-qi*Kc)

    # Get heading (Eq. 7)
    thetadoti = vi*Ki
    thetai = np.zeros(len(thetadoti))
    thetai[0] = pos0[2]
    for i in range(1, len(thetadoti)):
        thetai[i] = thetai[i - 1] + thetadoti[i - 1] * dt

    # Get follower velocities
    xdoti = vi*np.cos(thetai)
    ydoti = vi*np.sin(thetai)

    # Get follower positions
    # Initialize arrays to store positions
    xi = np.zeros(len(thetadoti))
    yi = np.zeros(len(thetadoti))

    xi[0] = pos0[0]
    yi[0] = pos0[1]

    # Integrate velocities to get positions
    for i in range(1, len(thetadoti)):
        xi[i] = xi[i - 1] + xdoti[i - 1] * dt
        yi[i] = yi[i - 1] + ydoti[i - 1] * dt
        thetai[i] = thetai[i - 1] + thetadoti[i - 1] * dt

    return xi, yi, thetai

def get_dynamic(pos0, vc, Kc, q, dq_ds, d2q_ds2, dt):

    # May be different lens due to differentiation methods
    trunc_idx = np.min([len(q), len(Kc), len(dq_ds)])
    q = q[0:trunc_idx]
    dq_ds = dq_ds[0:trunc_idx]
    d2q_ds2 = d2q_ds2[0:trunc_idx]
    Kc = Kc[0:trunc_idx]

    # Get Q (Eq. 13)
    Q = np.sqrt(dq_ds**2 + (1 - q*Kc)**2)

    vi = []
    Ki = []
    for i in range(len(Q)):
        # Get controls (Eq. 11)
        if (1 - q[i]*Kc[i]) < 0:
            v = -Q[i]*vc[i]
            K = -(1 / Q[i]) * (Kc[i] + ((1-q[i]*Kc[i])*d2q_ds2[i] + Kc[i]*(dq_ds[i])**2) / Q[i]**2)
        else:
            v = Q[i]*vc[i]
            K = (1 / Q[i]) * (Kc[i] + ((1-q[i]*Kc[i])*d2q_ds2[i] + Kc[i]*(dq_ds[i])**2) / Q[i]**2)

        vi.append(v)
        Ki.append(K)

    vi = np.array(vi)
    Ki = np.array(Ki) 

    # Get heading (Eq. 7)
    thetadoti = vi*Ki
    thetai = np.zeros(len(thetadoti))
    thetai[0] = pos0[2]
    for i in range(1, len(thetadoti)):
        thetai[i] = thetai[i - 1] + thetadoti[i - 1] * dt

    # Get follower velocities
    xdoti = vi*np.cos(thetai)
    ydoti = vi*np.sin(thetai)

    # Initialize and get follower positions
    xi = np.zeros(len(thetadoti))
    yi = np.zeros(len(thetadoti))

    xi[0] = pos0[0]
    yi[0] = pos0[1]


    # Integrate velocities to get positions
    for i in range(1, len(thetadoti)):
        xi[i] = xi[i - 1] + xdoti[i - 1] * dt
        yi[i] = yi[i - 1] + ydoti[i - 1] * dt
        thetai[i] = thetai[i - 1] + thetadoti[i - 1] * dt

    return xi, yi, thetai

