import numpy as np

# Kinematic model parameters
v_straight = 1.0                # Linear velocity during straight phase (m/s)
v_turn = 0.5                    # Linear velocity during turning phase (m/s)
curvature1 = 0.2                # Curvature of the turn (1/m)
omega1 = v_turn * curvature1    # Angular velocity during turning phase (rad/s)
curvature2 = -0.4               # Curvature of the turn (1/m)
omega2 = v_turn * curvature2    # Angular velocity during turning phase (rad/s)

# Time parameters
dt = 0.001           # Time step (s)

# Trajectory durations
straight_duration1 = 10.0                                   # Duration of straight phase (s)
turn_duration1 = abs(0.55 * np.pi / (v_turn * curvature1))  # Duration of turning phase (s)
straight_duration2 = 10.0                                   # Duration of straight phase (s)
turn_duration2 = abs(0.65 * np.pi / (v_turn * curvature2))  # Duration of turning phase (s)
straight_duration3 = 10.0                                   # Duration of straight phase (s)

# Plotting params
ARROW_LENGTH = 5
