"""
sim.py

This file contains the main simulation loop for the Game-induced
Nonlinear Opinion Dynamics (GiNOD) project.
"""

from copy import deepcopy
import numpy as np
import os

from constraints import BoxConstraint
from cost import *
from dynamic_systems import *
from ILQSolver import ILQSolver
from RHCPlanner import RHCPlanner

DATA_DIR = "./GiNOD/data/"

"""
State Vector: [p_x1, p_y1, phi_1, v_1, ... , p_xn, p_yn, phi_n, v_n]
- p_x: the x position of the center of the agent's rear axes
- p_y: the y position of the center of the agent's rear axes
- phi: the yaw angle with respect to the x-axis
- v: the velocity of the agent

Command Vector: [delta, a]
- delta: the steering angle
- a: the acceleration
"""

#################################################
#             Simulation Parameters
#################################################
n_a = 2                                 # Number of agents
n_th = np.full(n_a, 2)                  # Number of decision options for each agent
epsilon = 0.001                         # Small positive constant
method = "QMDPL0"                       # QMDP method (QMDPL0, QMDPL1, or QMDPL1L0)
n_sim = 130                             # Number of simulation steps

a_min = -2.0                            # Minimum acceleration
a_max = 1.0                             # Maximum acceleration
w_min = -0.1                            # Minimum steering angle
w_max = 0.1                             # Maximum steering angle
v_max = 4.0                             # Maximum velocity
v_min = 2.0                             # Minimum velocity
v_goal = 3.0                            # Goal velocity

goal_x_lb = 30.0                        # Goal x-axis lower bound
goal_1_y = 7.0                          # Goal 1 y-axis position
goal_2_y = 0.0                          # Goal 2 y-axis position
goal_1_w_p1 = 15.0                      # Goal 1 weight for player 1
goal_1_w_p2 = 15.0                      # Goal 1 weight for player 2
goal_2_w_p1 = 15.0                      # Goal 2 weight for player 1
goal_2_w_p2 = 15.0                      # Goal 2 weight for player 2

road_bounds = (-1.5, 8.5)               # Road boundary limits
toll_station_x_bounds = (50, 80)        # Toll station x-axis limits
toll_station_2_py = 3.5                 # Toll station 2 y-axis position
toll_station_width = 3.5                # Toll station width
proximity_threshold = 7.0               # Proximity threshold
ctrl_limit_slack_mult = 1.0             # Control limit slack multiplier

look_ahead = 5                          # Look-ahead horizon for QMDP
solver_max_iter = 35                    # Maximum number of iterations for ILQSolver
solver_time_res = 0.2                   # Time resolution for ILQSolver (s)
solver_alpha_scaling = 5                # Scaling factor for ILQSolver
solver_time_horizon = 3.0               # Time horizon for ILQSolver (s)
solver_verbose = False                  # Verbose output

t_horizon = 10.0                        # End Time
dt= 0.2                                 # Time Step
HORIZON_STEPS = int(t_horizon / dt)     # Horizon Step

#################################################
#               Initial Conditions
#################################################
x = np.array([0., 5., 3., 5., 5., 2., 0., 3.])  # Initial joint state vector
z = np.full(n_a, epsilon)                       # Initial opinion vector
z_bar = np.full(n_a, 1.0/n_a)                   # Initial nominal opinion vector
attn = np.full(n_a, 0.0)                        # Initial attention vector

#################################################
#           Create subsystem dynamics
#################################################
car_R = Car4D(l=2.7, T=dt)
car_H = Car4D(l=2.7, T=dt)

twoCar_casadi = TwoCar8D(l=2.7, T=dt)

jnt_sys = ProductMultiPlayerDynamicalSystem([car_R, car_H], T=dt)
x_dim = jnt_sys._x_dim

#################################################
#             Define Costs for Car R
#################################################
car_R_px_index = 0
car_R_py_index = 1
car_R_psi_index = 2
car_R_vel_index = 3
car_R_position_indices_in_product_state = (0, 1)

# Tracks the target heading.
car_R_goal_psi_cost = ReferenceDeviationCost(
    reference=0.0, dimension=car_R_psi_index, is_x=True, name="car_R_goal_psi",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

# Tracks the target velocity.
car_R_goal_vel_cost = ReferenceDeviationCost(
    reference=v_goal, dimension=car_R_vel_index, is_x=True, name="car_R_goal_vel",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

# Penalizes car speed above a threshold near the toll station.
car_R_maxv_cost = MaxVelCostPxDependent(
    v_index=car_R_vel_index, px_index=car_R_px_index, max_v=v_max,
    px_lb=toll_station_x_bounds[0], px_ub=toll_station_x_bounds[1], name="car_R_maxv",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

# Round boundary costs.
car_R_lower_road_cost = SemiquadraticCost(
    dimension=car_R_py_index, threshold=road_bounds[0], oriented_right=False,
    is_x=True, name="car_R_lower_road_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)
car_R_upper_road_cost = SemiquadraticCost(
    dimension=car_R_py_index, threshold=road_bounds[1], oriented_right=True,
    is_x=True, name="car_R_upper_road_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

# Minimum velocity constraint.
car_R_min_vel_cost = SemiquadraticCost(
    dimension=car_R_vel_index, threshold=v_min, oriented_right=False, is_x=True,
    name="car_R_min_vel_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

# Control costs.
car_R_a_cost = QuadraticCost(0, 0.0, False, "car_R_a_cost", HORIZON_STEPS, x_dim, car_R._u_dim)
car_R_w_cost = QuadraticCost(
    1, 0.0, False, "car_R_w_cost", HORIZON_STEPS, x_dim, car_R._u_dim
)

# Control constraint costs.
car_R_a_constr_cost = BoxInputConstraintCost(
    0, ctrl_limit_slack_mult * a_min, ctrl_limit_slack_mult * a_max, q1=1., q2=5.,
    name="car_R_a_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)
car_R_w_constr_cost = BoxInputConstraintCost(
    1, ctrl_limit_slack_mult * w_min, ctrl_limit_slack_mult * w_max, q1=1., q2=5.,
    name="car_R_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)


#################################################
#           Define Costs for Car H
#################################################
car_H_px_index = 4
car_H_py_index = 5
car_H_psi_index = 6
car_H_vel_index = 7
car_H_position_indices_in_product_state = (4, 5)

# Tracks the target heading.
car_H_goal_psi_cost = ReferenceDeviationCost(
    reference=0.0, dimension=car_H_psi_index, is_x=True, name="car_H_goal_psi",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

# Tracks the target velocity.
car_H_goal_vel_cost = ReferenceDeviationCost(
    reference=v_goal, dimension=car_H_vel_index, is_x=True, name="car_H_goal_vel",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)

# Penalizes car speed above a threshold near the toll station.
car_H_maxv_cost = MaxVelCostPxDependent(
    v_index=car_H_vel_index, px_index=car_H_px_index, max_v=v_max,
    px_lb=toll_station_x_bounds[0], px_ub=toll_station_x_bounds[1], name="car_H_maxv",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

# Round boundary costs.
car_H_lower_road_cost = SemiquadraticCost(
    dimension=car_H_py_index, threshold=road_bounds[0], oriented_right=False,
    is_x=True, name="car_H_lower_road_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)
car_H_upper_road_cost = SemiquadraticCost(
    dimension=car_H_py_index, threshold=road_bounds[1], oriented_right=True,
    is_x=True, name="car_H_upper_road_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)

# Minimum velocity constraint.
car_H_min_vel_cost = SemiquadraticCost(
    dimension=car_H_vel_index, threshold=v_min, oriented_right=False, is_x=True,
    name="car_H_min_vel_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)

# Control costs.
car_H_a_cost = QuadraticCost(0, 0.0, False, "car_H_a_cost", HORIZON_STEPS, x_dim, car_H._u_dim)
car_H_w_cost = QuadraticCost(
    1, 0.0, False, "car_H_w_cost", HORIZON_STEPS, x_dim, car_H._u_dim
)

# Control constraint costs.
car_H_a_constr_cost = BoxInputConstraintCost(
    0, ctrl_limit_slack_mult * a_min, ctrl_limit_slack_mult * a_max, q1=1., q2=5.,
    name="car_H_a_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)
car_H_w_constr_cost = BoxInputConstraintCost(
    1, ctrl_limit_slack_mult * w_min, ctrl_limit_slack_mult * w_max, q1=1., q2=5.,
    name="car_H_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)


#################################################
#               Proximity Cost
#################################################
proximity_cost_RH = ProductStateProximityCostTwoPlayer([
    car_R_position_indices_in_product_state,
    car_H_position_indices_in_product_state,
], proximity_threshold, "proximity", HORIZON_STEPS, x_dim, car_R._u_dim)


#################################################
#   Build up total costs (intent-independent)
#   for both players.
#################################################
car_R_player_id = 1
car_R_cost = PlayerCost()
car_R_cost.add_cost(car_R_goal_psi_cost, "x", 1.0)
car_R_cost.add_cost(car_R_goal_vel_cost, "x", 1.0)

car_R_cost.add_cost(car_R_lower_road_cost, "x", 50.0)
car_R_cost.add_cost(car_R_upper_road_cost, "x", 50.0)
car_R_cost.add_cost(car_R_min_vel_cost, "x", 50.0)
car_R_cost.add_cost(proximity_cost_RH, "x", 150.0)

car_R_cost.add_cost(car_R_w_cost, car_R_player_id, 10.0)
car_R_cost.add_cost(car_R_a_cost, car_R_player_id, 1.0)

car_R_cost.add_cost(car_R_w_constr_cost, car_R_player_id, 50.0)
car_R_cost.add_cost(car_R_a_constr_cost, car_R_player_id, 50.0)


car_H_player_id = 2
car_H_cost = PlayerCost()
car_H_cost.add_cost(car_H_goal_psi_cost, "x", 1.0)
car_H_cost.add_cost(car_H_goal_vel_cost, "x", 1.0)

car_H_cost.add_cost(car_H_lower_road_cost, "x", 50.0)
car_H_cost.add_cost(car_H_upper_road_cost, "x", 50.0)
car_H_cost.add_cost(car_H_min_vel_cost, "x", 50.0)
car_H_cost.add_cost(proximity_cost_RH, "x", 150.0)

car_H_cost.add_cost(car_H_w_cost, car_H_player_id, 10.0)
car_H_cost.add_cost(car_H_a_cost, car_H_player_id, 1.0)

car_H_cost.add_cost(car_H_w_constr_cost, car_H_player_id, 50.0)
car_H_cost.add_cost(car_H_a_constr_cost, car_H_player_id, 50.0)


#################################################
#           Toll station avoidance cost
#################################################
ts_px = toll_station_x_bounds[0]
ts_py = toll_station_2_py
while ts_px < toll_station_x_bounds[1]:
  car_R_toll_station_cost_tmp = ProximityCost(
      position_indices=car_R_position_indices_in_product_state, point_px=ts_px, point_py=ts_py,
      max_distance=toll_station_width, name="", horizon=HORIZON_STEPS, x_dim=x_dim,
      ui_dim=car_R._u_dim
  )
  car_R_cost.add_cost(car_R_toll_station_cost_tmp, "x", 150.0)

  car_H_toll_station_cost_tmp = ProximityCost(
      position_indices=car_H_position_indices_in_product_state, point_px=ts_px, point_py=ts_py,
      max_distance=toll_station_width, name="", horizon=HORIZON_STEPS, x_dim=x_dim,
      ui_dim=car_H._u_dim
  )
  car_H_cost.add_cost(car_H_toll_station_cost_tmp, "x", 150.0)

  ts_px += toll_station_width


#################################################
#           Define Input Constraints
#################################################
u_constraints_car_R = BoxConstraint(
    lower=jnp.hstack((a_min, w_min)), upper=jnp.hstack((a_max, w_max))
)
u_constraints_car_H = BoxConstraint(
    lower=jnp.hstack((a_min, w_min)), upper=jnp.hstack((a_max, w_max))
)


#################################################
#               Intialize Strategies
#################################################
car_R_Ps = jnp.zeros((car_R._u_dim, jnt_sys._x_dim, HORIZON_STEPS))
car_H_Ps = jnp.zeros((car_H._u_dim, jnt_sys._x_dim, HORIZON_STEPS))

car_R_alphas = jnp.zeros((car_R._u_dim, HORIZON_STEPS))
car_H_alphas = jnp.zeros((car_H._u_dim, HORIZON_STEPS))


#################################################
#       Set up intent-dependent costs
#################################################
car_R_tgt_booth_cost_1 = ReferenceDeviationCostPxDependent(
    reference=goal_1_y, dimension=car_R_py_index, px_dim=car_R_px_index,
    px_lb=goal_x_lb, name="car_R_tgt_booth_cost_1", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)
car_R_tgt_booth_cost_2 = ReferenceDeviationCostPxDependent(
    reference=goal_2_y, dimension=car_R_py_index, px_dim=car_R_px_index,
    px_lb=goal_x_lb, name="car_R_tgt_booth_cost_2", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)
car_H_tgt_booth_cost_1 = ReferenceDeviationCostPxDependent(
    reference=goal_1_y, dimension=car_H_py_index, px_dim=car_H_px_index,
    px_lb=goal_x_lb, name="car_H_tgt_booth_cost_1", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)
car_H_tgt_booth_cost_2 = ReferenceDeviationCostPxDependent(
    reference=goal_2_y, dimension=car_H_py_index, px_dim=car_H_px_index,
    px_lb=goal_x_lb, name="car_H_tgt_booth_cost_2", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)

car_R_cost_subgame11 = deepcopy(car_R_cost)
car_H_cost_subgame11 = deepcopy(car_H_cost)
car_R_cost_subgame11.add_cost(car_R_tgt_booth_cost_1, "x", goal_1_w_p1)
car_H_cost_subgame11.add_cost(car_H_tgt_booth_cost_1, "x", goal_1_w_p2)

car_R_cost_subgame12 = deepcopy(car_R_cost)
car_H_cost_subgame12 = deepcopy(car_H_cost)
car_R_cost_subgame12.add_cost(car_R_tgt_booth_cost_1, "x", goal_1_w_p1)
car_H_cost_subgame12.add_cost(car_H_tgt_booth_cost_2, "x", goal_2_w_p2)

car_R_cost_subgame21 = deepcopy(car_R_cost)
car_H_cost_subgame21 = deepcopy(car_H_cost)
car_R_cost_subgame21.add_cost(car_R_tgt_booth_cost_2, "x", goal_2_w_p1)
car_H_cost_subgame21.add_cost(car_H_tgt_booth_cost_1, "x", goal_1_w_p2)

car_R_cost_subgame22 = deepcopy(car_R_cost)
car_H_cost_subgame22 = deepcopy(car_H_cost)
car_R_cost_subgame22.add_cost(car_R_tgt_booth_cost_2, "x", goal_2_w_p1)
car_H_cost_subgame22.add_cost(car_H_tgt_booth_cost_2, "x", goal_2_w_p2)


#################################################
#       Setup ILQSolvers for each subgame
#################################################
alpha_scaling = np.linspace(0.01, 2.0, solver_alpha_scaling)

solver11 = ILQSolver(
    jnt_sys, [car_R_cost_subgame11, car_H_cost_subgame11], [car_R_Ps, car_H_Ps],
    [car_R_alphas, car_H_alphas], alpha_scaling, solver_max_iter,
    u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=solver_verbose, name="subgame_11"
)
solver12 = ILQSolver(
    jnt_sys, [car_R_cost_subgame12, car_H_cost_subgame12], [car_R_Ps, car_H_Ps],
    [car_R_alphas, car_H_alphas], alpha_scaling, solver_max_iter,
    u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=solver_verbose, name="subgame_12"
)
solver21 = ILQSolver(
    jnt_sys, [car_R_cost_subgame21, car_H_cost_subgame21], [car_R_Ps, car_H_Ps],
    [car_R_alphas, car_H_alphas], alpha_scaling, solver_max_iter,
    u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=solver_verbose, name="subgame_21"
)
solver22 = ILQSolver(
    jnt_sys, [car_R_cost_subgame22, car_H_cost_subgame22], [car_R_Ps, car_H_Ps],
    [car_R_alphas, car_H_alphas], alpha_scaling, solver_max_iter,
    u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=solver_verbose, name="subgame_22"
)

subgames = [[solver11, solver12], [solver21, solver22]]


#################################################
#               Initialize GiNOD
#################################################
GiNOD = NonlinearOpinionDynamicsTwoPlayer(
    x_indices_P1=np.array((0, 1, 2, 3)),
    x_indices_P2=np.array((4, 5, 6, 7)),
    z_indices_P1=np.array((8, 9)),
    z_indices_P2=np.array((10, 11)),
    att_indices_P1=np.array((12,)),
    att_indices_P2=np.array((13,)),
    z_P1_bias=0. * np.ones((2,)),
    z_P2_bias=0. * np.ones((2,)),
    T=dt,
    damping_opn=0.1,
    damping_att=[0.5, 0.5],
    rho=[0.8, 0.8],
)


#################################################
#           Run Simulation Planner
#################################################
car_R_px0 = 0.0
car_R_py0 = 5.0
car_R_theta0 = 0.0
car_R_v0 = 3.0
car_R_x0 = np.array([car_R_px0, car_R_py0, car_R_theta0, car_R_v0])

car_H_px0 = 5.0
car_H_py0 = 2.0
car_H_theta0 = 0.0
car_H_v0 = 3.0
car_H_x0 = np.array([car_H_px0, car_H_py0, car_H_theta0, car_H_v0])

jnt_x0 = np.concatenate([car_R_x0, car_H_x0], axis=0)
z0 = np.array(([1e-2, -1e-2, -1e-2, 1e-2, 0., 0.]))

W_R = np.diag([5.0, 5.0])
W_H = np.diag([5.0, 5.0])

planner = RHCPlanner(
    subgames, n_sim, jnt_sys, twoCar_casadi, GiNOD, look_ahead,
    [a_min, a_max], [w_min, w_max], method='QMDPL0', W_ctrl=[W_R, W_H]
)
planner.plan(jnt_x0, z0)

#################################################
#                   Save Data
#################################################
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

np.save(f'{DATA_DIR}/test_L0_t1_xs.npy', planner.xs)
np.save(f'{DATA_DIR}/test_L0_t1_zs.npy', planner.zs)
np.save(f'{DATA_DIR}/test_L0_t1_Hs.npy', planner.Hs)
np.save(f'{DATA_DIR}/test_L0_t1_PoI.npy', planner.PoI)
