"""
dynamics_systems.py

This file contains all base classes and child classes for dynamic systems
in the GiNOD project.
"""

from functools import partial
from typing import Tuple

import numpy as np
from casadi import vertcat, horzcat, kron
import jax
from jax import jit, jacfwd, lax
import jax.numpy as jnp
from jaxlib.xla_extension import ArrayImpl

from utils import softmax


class MultiPlayerDynamicalSystem(object):
    """
    Base class for all multiplayer continuous-time dynamical systems. Supports
    numrical integration and linearization.
    """

    def __init__(self, x_dim, u_dims, T=0.1):
        """
        Initializer.

        Args:
        - x_dim (int): number of state dimensions
        - u_dims ([int]): liset of number of control dimensions for each player
        - T (float): time interval
        """

        self._x_dim = x_dim
        self._u_dims = u_dims
        self._T = T
        self._num_players = len(u_dims)

        # Pre-compute Jacobian matrices.
        self.jac_f = jit(jacfwd(self.disc_time_dyn, argnums=[0, 1]))

    @partial(jit, static_argnums=(0,))
    def cont_time_dyn(self, x: ArrayImpl, u_list: list, k: int = 0, args=()) -> list:
        """
        Computes the time derivative of state for a particular state/control.

        Args:
        - x (ArrayImpl): joint state (nx,)
        - u_list (list of ArrayImpl): list of controls [(nu_0,), (nu_1,), ...]

        Returns:
        - list of ArrayImpl: list of next states [(nx_0,), (nx_1,), ...]
        """
        
        raise NotImplementedError("cont_time_dyn() has not been implemented.")

    @partial(jit, static_argnums=(0,))
    def disc_time_dyn(self, x0: ArrayImpl, u0_list: list, k: int = 0, args=()) -> list:
        """
        Computes the one-step evolution of the system in discrete time with Euler
        integration.

        Args:
        - x0 (ArrayImpl): joint state (nx,)
        - u0_list (list of ArrayImpl): list of controls [(nu_0,), (nu_1,), ...]

        Returns:
        - list of ArrayImpl: list of next states [(nx_0,), (nx_1,), ...]
        """

        x_dot = self.cont_time_dyn(x0, u0_list, k, args)
        return x0 + self._T * x_dot

    @partial(jit, static_argnums=(0,))
    def linearize_discrete_jitted(self, x0: ArrayImpl, u0_list: list, k: int = 0,
                                  args=()) -> Tuple[ArrayImpl, list]:
        """
        Compute the Jacobian linearization of the dynamics for a particular
        state `x0` and control `u0`. Outputs `A` and `B` matrices of a
        discrete-time linear system:
                ``` x(k + 1) - x0 = A (x(k) - x0) + B (u(k) - u0) ```

        Args:
        - x0 (ArrayImpl): joint state (nx,)
        - u0_list (list of ArrayImpl): list of controls [(nu_0,), (nu_1,), ...]

        Returns:
        - ArrayImpl: the Jacobian of next state w.r.t. x0.
        - list of ArrayImpl: the Jacobian of next state w.r.t. u0_i.
        """

        A_disc, B_disc = self.jac_f(x0, u0_list, k, args)
        return A_disc, B_disc


class ProductMultiPlayerDynamicalSystem(MultiPlayerDynamicalSystem):
    """
    Implements a multiplayer dynamical system who's dynamics decompose into a Cartesian
    product of single-player dynamical systems.
    """

    def __init__(self, subsystems, T=0.1):
        """
        Initializer.

        Args:
        - subsystems ([DynamicalSystem]): list of component (single-player)
            dynamical systems
        - T (float): time interval
        """
        
        self._subsystems = subsystems
        self._x_dims = [sys._x_dim for sys in subsystems]

        x_dim = sum(self._x_dims)
        self._x_dim = x_dim
        u_dims = [sys._u_dim for sys in subsystems]
        self.u_dims = u_dims

        super(ProductMultiPlayerDynamicalSystem, self).__init__(x_dim, u_dims, T)

        self.update_lifting_matrices()
        self._num_opn_dyn = 0

    def update_lifting_matrices(self):
        """
        Updates the lifting matrices.
        """

        # Create lifting matrices LMx_i for subsystem i such that LMx_i @ x = xi.
        _split_index = np.hstack((0, np.cumsum(np.asarray(self._x_dims))))
        self._LMx = [np.zeros((xi_dim, self._x_dim)) for xi_dim in self._x_dims]

        for i in range(len(self._x_dims)):
            self._LMx[i][:, _split_index[i]:_split_index[i + 1]] = np.eye(self._x_dims[i])
            self._LMx[i] = jnp.asarray(self._LMx[i])

        # Create lifting matrices LMu_i for subsystem i such that LMu_i @ u = ui.
        u_dims = self.u_dims
        u_dim = sum(u_dims)
        _split_index = np.hstack((0, np.cumsum(np.asarray(u_dims))))
        self._LMu = [np.zeros((ui_dim, u_dim)) for ui_dim in u_dims]
        for i in range(self._num_players):
            self._LMu[i][:, _split_index[i]:_split_index[i + 1]] = np.eye(u_dims[i])
            self._LMu[i] = jnp.asarray(self._LMu[i])

    def add_opinion_dyn(self, opn_dyns):
        """
        Append the physical subsystems with opinion dynamics, which do not have
        controls but *should* be affected by the physical states.

        Args:
        - opn_dyns (DynamicalSystem): opinion dynamics
        """

        opn_dyns._start_index = self._x_dim
        self._subsystems.append(opn_dyns)
        self._num_opn_dyn += 1

        self._x_dim += opn_dyns._x_dim
        self._x_dims.append(opn_dyns._x_dim)

        self.update_lifting_matrices()
        self._LMx += [jnp.eye(self._x_dim)] * self._num_opn_dyn

    @partial(jit, static_argnums=(0,))
    def cont_time_dyn(self, x: ArrayImpl, u_list: list, k: int = 0, args=()) -> list:
        """
        Computes the time derivative of state for a particular state/control.

        Args:
        - x (ArrayImpl): joint state (nx,)
        - u_list (list of ArrayImpl): list of controls [(nu_0,), (nu_1,), ...]

        Returns:
        - list of ArrayImpl: list of next states [(nx_0,), (nx_1,), ...]
        """

        u_list += [None] * self._num_opn_dyn
        x_dot_list = [
            subsys.cont_time_dyn(LMx @ x, u0, k, args)
            for subsys, LMx, u0 in zip(self._subsystems, self._LMx, u_list)
        ]

        return jnp.concatenate(x_dot_list, axis=0)


class DynamicalSystem(object):
    """
    Base class for all continuous-time dynamical systems. Supports numerical
    integration and linearization.
    """

    def __init__(self, x_dim, u_dim, T=0.1):
        """
        Initializer.

        Args:
        - x_dim (int): number of state dimensions
        - u_dim (int): number of control dimensions
        - T (float): time interval
        """

        self._x_dim = x_dim
        self._u_dim = u_dim
        self._T = T

    @partial(jit, static_argnums=(0,))
    def cont_time_dyn(self, x0: ArrayImpl, u0: ArrayImpl, k: int = 0, *args) -> ArrayImpl:
        """
        Abstract method.
        Computes the time derivative of state for a particular state/control.

        Args:
        - x0 (ArrayImpl): (nx,)
        - u0 (ArrayImpl): (nu,)

        Returns:
        - ArrayImpl: next state (nx,)
        """

        raise NotImplementedError("cont_time_dyn() has not been implemented.")

    @partial(jit, static_argnums=(0,))
    def disc_time_dyn(self, x0: ArrayImpl, u0: ArrayImpl, k: int = 0, args=()) -> ArrayImpl:
        """
        Computes the one-step evolution of the system in discrete time with Euler
        integration.

        Args:
        - x0 (ArrayImpl): (nx,)
        - u0 (ArrayImpl): (nu,)

        Returns:
        - ArrayImpl: next state (nx,)
        """

        x_dot = self.cont_time_dyn(x0, u0, k, args)

        return x0 + self._T * x_dot

    @partial(jit, static_argnums=(0,))
    def linearize_discrete_jitted(self, x0: ArrayImpl, u0: ArrayImpl, k: int = 0,
                                args=()) -> Tuple[ArrayImpl, ArrayImpl]:
        """
        Compute the Jacobian linearization of the dynamics for a particular
        state `x0` and control `u0`. Outputs `A` and `B` matrices of a
        discrete-time linear system:
                ``` x(k + 1) - x0 = A (x(k) - x0) + B (u(k) - u0) ```

        Args:
        - x0 (ArrayImpl): (nx,)
        - u0 (ArrayImpl): (nu,)

        Returns:
        - ArrayImpl: the Jacobian of next state w.r.t. the current state.
        - ArrayImpl: the Jacobian of next state w.r.t. the current control.
        """

        A_disc, B_disc = self.jac_f(x0, u0, k, args)

        return A_disc, B_disc


class NonlinearOpinionDynamicsTwoPlayer(DynamicalSystem):
    """
    Two Player Nonlinear Opinion Dynamics.

    For jit compatibility, number of players is hardcoded to 2 to avoid loops.

    Joint state vector should be organized as
        xi := [x, z_P1, z_P2, lambda_P1, lambda_P2]
    """

    def __init__(self, x_indices_P1, x_indices_P2, z_indices_P1, z_indices_P2,
                 att_indices_P1, att_indices_P2,z_P1_bias, z_P2_bias, damping_opn=0.0,
                 damping_att=[0.0, 0.0], rho=[1.0, 1.0], T=0.1,z_norm_thresh=10.0):
        """
        Initializer.

        Args:
        - x_indices_P1 (ArrayImpl, dtype=int32): P1 x (physical states) indices
        - x_indices_P2 (ArrayImpl, dtype=int32): P2 x (physical states) indices
        - z_indices_P1 (ArrayImpl, dtype=int32): P1 z (opinion states) indices
        - z_indices_P2 (ArrayImpl, dtype=int32): P2 z (opinion states) indices
        - att_indices_P1 (ArrayImpl, dtype=int32): P1 attention indices
        - att_indices_P2 (ArrayImpl, dtype=int32): P2 attention indices
        - z_P1_bias (ArrayImpl): (nz,) P1 opinion state bias
        - z_P2_bias (ArrayImpl): (nz,) P2 opinion state bias
        - damping_opn (float, optional): z damping parameter. Defaults to 0.0.
        - damping_att (float, optional): att damping parameter. Defaults to 0.0.
        - rho (float, optional): att scaling parameter. Defaults to 1.0.
        - T (float, optional): time interval. Defaults to 0.1.
        """

        self._x_indices_P1 = x_indices_P1
        self._x_indices_P2 = x_indices_P2
        self._z_indices_P1 = z_indices_P1
        self._z_indices_P2 = z_indices_P2
        self._att_indices_P1 = att_indices_P1
        self._att_indices_P2 = att_indices_P2
        self._z_P1_bias = z_P1_bias
        self._z_P2_bias = z_P2_bias
        self._damping_opn = damping_opn
        self._damping_att = damping_att
        self._rho = rho

        self._eps = 0.
        self._PoI_max = 10.0
        self._z_norm_thresh = z_norm_thresh

        # Players' number of options
        self._num_opn_P1 = len(self._z_indices_P1)
        self._num_opn_P2 = len(self._z_indices_P2)

        self._num_att_P1 = len(self._att_indices_P1)
        self._num_att_P2 = len(self._att_indices_P2)

        self._x_dim = (self._num_opn_P1 + self._num_opn_P2 + self._num_att_P1 + self._num_att_P2)

        super(NonlinearOpinionDynamicsTwoPlayer, self).__init__(self._x_dim, 0, T)

    @partial(jit, static_argnums=(0,))
    def cont_time_dyn(self, x: ArrayImpl, ctrl=None, subgame: Tuple = ()) -> ArrayImpl:
        """
        Computes the time derivative of state for a particular state/control.
        This is an autonomous system.

        Args:
        - x (ArrayImpl): (nx,) where nx is the dimension of the joint
            system (physical subsystems plus all players' opinion dynamics)
            For each opinion dynamics, their state := (z, u) where z is the
            opinion state and u is the attention parameter
        - ctrl (ArrayImpl): (nu,) control input
        - subgame (Tuple): subgame information

        Returns:
            ArrayImpl: next state (nx,)
            ArrayImpl: H (nz, nz)
        """

        def Vhat1(z1: ArrayImpl, z2: ArrayImpl, x_ph: ArrayImpl) -> ArrayImpl:
            """
            Opinion-weighted game value function for P1.

            Args:
            - z1 (ArrayImpl): (nz,) P1 opinion state
            - z2 (ArrayImpl): (nz,) P2 opinion state
            - x_ph (ArrayImpl): (nx,) physical state
            """

            V_hat = 0.
            for l1 in range(self._num_opn_P1):
                for l2 in range(self._num_opn_P2):
                    xe = x_ph - x_ph_nom[:, l1, l2]
                    Z_sub = Z_P1[:, :, l1, l2]
                    zeta_sub = zeta_P1[:, l1, l2]
                    nom_cost_sub = nom_cost_P1[l1, l2]
                    V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
                    V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub

            return V_hat

        def Vhat2(z1: ArrayImpl, z2: ArrayImpl, x_ph: ArrayImpl) -> ArrayImpl:
            """
            Opinion-weighted game value function for P2.
            
            Args:
            - z1 (ArrayImpl): (nz,) P1 opinion state
            - z2 (ArrayImpl): (nz,) P2 opinion state
            - x_ph (ArrayImpl): (nx,) physical state
            """

            V_hat = 0.
            for l1 in range(self._num_opn_P1):
                for l2 in range(self._num_opn_P2):
                    xe = x_ph - x_ph_nom[:, l1, l2]
                    Z_sub = Z_P2[:, :, l1, l2]
                    zeta_sub = zeta_P2[:, l1, l2]
                    nom_cost_sub = nom_cost_P2[l1, l2]
                    V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
                    V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub

            return V_hat

        def compute_PoI_P1(z1: ArrayImpl, x_ph: ArrayImpl) -> ArrayImpl:
            """
            Computes the Price of Indecision (PoI) for P1.

            Args:
            - z1 (ArrayImpl): (nz,) P1 opinion state
            - x_ph (ArrayImpl): (nx,) physical state
            """

            ratios = jnp.zeros((self._num_opn_P2,))

            # Outer loop over P2's (opponent) options.
            for l2 in range(self._num_opn_P2):
                V_subs = jnp.zeros((self._num_opn_P1,))

                # Inner loop over P1's (ego) options.
                for l1 in range(self._num_opn_P1):
                    xe = x_ph - x_ph_nom[:, l1, l2]  # error state
                    Z_sub = Z_P1[:, :, l1, l2]
                    zeta_sub = zeta_P1[:, l1, l2]
                    nom_cost_sub = nom_cost_P1[l1, l2]
                    V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
                    V_subs = V_subs.at[l1].set(V_sub)

                # Normalize to avoid large numbers.
                V_subs = jax.nn.softmax(jax.nn.standardize(V_subs))

                numer = 0.
                for l1 in range(self._num_opn_P1):
                    numer += softmax(z1, l1) * V_subs[l1]

                denom = jnp.min(V_subs)

                ratio = (numer + self._eps) / (denom + self._eps)
                ratios = ratios.at[l2].set(ratio)

            PoI = jnp.max(ratios)
            return jnp.minimum(PoI, self._PoI_max)

        def compute_PoI_P2(z2: ArrayImpl, x_ph: ArrayImpl) -> ArrayImpl:
            """
            Computes the Price of Indecision (PoI) for P2.

            Args:
            - z2 (ArrayImpl): (nz,) P2 opinion state
            - x_ph (ArrayImpl): (nx,) physical state
            """

            ratios = jnp.zeros((self._num_opn_P1,))

            # Outer loop over P1's (opponent) options.
            for l1 in range(self._num_opn_P1):
                numer = 0.
                V_subs = jnp.zeros((self._num_opn_P2,))

                # Inner loop over P2's (ego) options.
                for l2 in range(self._num_opn_P2):
                    xe = x_ph - x_ph_nom[:, l1, l2]  # error state
                    Z_sub = Z_P2[:, :, l1, l2]
                    zeta_sub = zeta_P2[:, l1, l2]
                    nom_cost_sub = nom_cost_P2[l1, l2]
                    V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
                    V_subs = V_subs.at[l2].set(V_sub)

                # Normalize to avoid large numbers.
                V_subs = jax.nn.softmax(jax.nn.standardize(V_subs))

                numer = 0.
                for l2 in range(self._num_opn_P2):
                    numer += softmax(z2, l2) * V_subs[l2]

                denom = jnp.min(V_subs)

                ratio = (numer + self._eps) / (denom + self._eps)
                ratios = ratios.at[l1].set(ratio)

            PoI = jnp.max(ratios)
            return jnp.minimum(PoI, self._PoI_max)

        def true_fn(PoI):
            return PoI

        def false_fn(PoI):
            return 1.0

        (Z_P1, Z_P2, zeta_P1, zeta_P2, x_ph_nom, znom_P1, znom_P2, nom_cost_P1, nom_cost_P2) = subgame

        # State variables.
        x_ph1 = x[self._x_indices_P1]
        x_ph2 = x[self._x_indices_P2]
        x_ph = jnp.hstack((x_ph1, x_ph2))

        z1 = x[self._z_indices_P1]
        z2 = x[self._z_indices_P2]
        z = jnp.hstack((z1, z2))

        att1 = x[self._att_indices_P1]
        att2 = x[self._att_indices_P2]

        # Compute game Hessians.
        dVhat1_dz1 = jacfwd(Vhat1, argnums=0)
        H1s = jacfwd(dVhat1_dz1, argnums=[0, 1])
        H1 = jnp.hstack(H1s(znom_P1, znom_P2, x_ph))

        dVhat2_dz2 = jacfwd(Vhat2, argnums=1)
        H2s = jacfwd(dVhat2_dz2, argnums=[0, 1])
        H2 = jnp.hstack(H2s(znom_P1, znom_P2, x_ph))

        # Normalize to avoid large numbers.
        H1 = jax.nn.standardize(H1)
        H2 = jax.nn.standardize(H2)

        # Compute the opinion state time derivative.
        att_1_vec = att1 * jnp.ones((self._num_opn_P1,))
        att_2_vec = att2 * jnp.ones((self._num_opn_P2,))

        D = jnp.diag(self._damping_opn * jnp.ones(self._num_opn_P1 + self._num_opn_P2,))

        H1z = att_1_vec * jnp.tanh(H1@z + self._z_P1_bias)
        H2z = att_2_vec * jnp.tanh(H2@z + self._z_P2_bias)

        z_dot = -D @ z + jnp.hstack((H1z, H2z))

        # Compute the attention time derivative.
        PoI_1 = jnp.nan_to_num(compute_PoI_P1(z1, x_ph), nan=1.0)
        PoI_2 = jnp.nan_to_num(compute_PoI_P2(z2, x_ph), nan=1.0)

        z1_norm = jnp.linalg.norm(z1)
        PoI_1 = lax.cond(z1_norm <= self._z_norm_thresh, true_fn, false_fn, PoI_1)

        z2_norm = jnp.linalg.norm(z2)
        PoI_2 = lax.cond(z2_norm <= self._z_norm_thresh, true_fn, false_fn, PoI_2)

        att1_dot = -self._damping_att[0] * att1 + self._rho[0] * (PoI_1-1.0)
        att2_dot = -self._damping_att[1] * att2 + self._rho[1] * (PoI_2-1.0)

        # Joint state time derivative.
        x_jnt_dot = jnp.hstack((z_dot, att1_dot, att2_dot))

        return x_jnt_dot, jnp.vstack((H1, H2)), PoI_1, PoI_2

    @partial(jit, static_argnums=(0,))
    def disc_dyn_jitted(self, x_ph: ArrayImpl, z1: ArrayImpl, z2: ArrayImpl,
                        att1: ArrayImpl, att2: ArrayImpl, ctrl=None,
                        subgame: Tuple = ()) -> ArrayImpl:
        """
        Computes the next opinion state.
        This is an autonomous system.

        Args:
        - x_ph (ArrayImpl): (nx,) physical state
        - z1 (ArrayImpl): (nz,) P1 opinion state
        - z2 (ArrayImpl): (nz,) P2 opinion state
        - att1 (ArrayImpl): (na,) P1 attention parameter
        - att2 (ArrayImpl): (na,) P2 attention parameter
        - ctrl (ArrayImpl): (nu,) control input
        - subgame (Tuple): subgame information
        """

        def Vhat1(z1: ArrayImpl, z2: ArrayImpl, x_ph: ArrayImpl) -> ArrayImpl:
            """
            Opinion-weighted game value function for P1.

            Args:
            - z1 (ArrayImpl): (nz,) P1 opinion state
            - z2 (ArrayImpl): (nz,) P2 opinion state
            - x_ph (ArrayImpl): (nx,) physical state
            """

            V_hat = 0.
            for l1 in range(self._num_opn_P1):
                for l2 in range(self._num_opn_P2):
                    xe = x_ph - x_ph_nom[:, l1, l2]
                    Z_sub = Z_P1[:, :, l1, l2]
                    zeta_sub = zeta_P1[:, l1, l2]
                    nom_cost_sub = nom_cost_P1[l1, l2]
                    V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
                    V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub

            return V_hat

        def Vhat2(z1: ArrayImpl, z2: ArrayImpl, x_ph: ArrayImpl) -> ArrayImpl:
            """
            Opinion-weighted game value function for P2.
            
            Args:
            - z1 (ArrayImpl): (nz,) P1 opinion state
            - z2 (ArrayImpl): (nz,) P2 opinion state
            - x_ph (ArrayImpl): (nx,) physical state
            """

            V_hat = 0.
            for l1 in range(self._num_opn_P1):
                for l2 in range(self._num_opn_P2):
                    xe = x_ph - x_ph_nom[:, l1, l2]
                    Z_sub = Z_P2[:, :, l1, l2]
                    zeta_sub = zeta_P2[:, l1, l2]
                    nom_cost_sub = nom_cost_P2[l1, l2]
                    V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
                    V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub

            return V_hat

        (Z_P1, Z_P2, zeta_P1, zeta_P2, x_ph_nom, znom_P1, znom_P2, nom_cost_P1, nom_cost_P2) = subgame

        # State variables.
        z = jnp.hstack((z1, z2))

        # Compute game Hessians.
        dVhat1_dz1 = jacfwd(Vhat1, argnums=0)
        H1s = jacfwd(dVhat1_dz1, argnums=[0, 1])
        H1 = jnp.hstack(H1s(znom_P1, znom_P2, x_ph))

        dVhat2_dz2 = jacfwd(Vhat2, argnums=1)
        H2s = jacfwd(dVhat2_dz2, argnums=[0, 1])
        H2 = jnp.hstack(H2s(znom_P1, znom_P2, x_ph))

        # Normalize to avoid large numbers.
        H1 = jax.nn.standardize(H1)
        H2 = jax.nn.standardize(H2) 

        # Compute the opinion state time derivative.
        att_1_vec = att1 * jnp.ones((self._num_opn_P1,))
        att_2_vec = att2 * jnp.ones((self._num_opn_P2,))

        D = jnp.diag(self._damping_opn * jnp.ones(self._num_opn_P1 + self._num_opn_P2,))

        H1z = att_1_vec * jnp.tanh(H1@z + self._z_P1_bias)
        H2z = att_2_vec * jnp.tanh(H2@z + self._z_P2_bias)

        z_dot = -D @ z + jnp.hstack((H1z, H2z))

        return z + self._T * z_dot

    def disc_dyn_two_player_casadi(self, x_ph, z1, z2, att1, att2, ctrl=None, subgame: Tuple = ()):
        """
        Computes the next opinion state using the analytical GiNOD.
        For optimization in casadi.

        Args:
        - x_ph (ArrayImpl): (nx,) physical state
        - z1 (ArrayImpl): (nz,) P1 opinion state
        - z2 (ArrayImpl): (nz,) P2 opinion state
        - att1 (ArrayImpl): (na,) P1 attention parameter
        - att2 (ArrayImpl): (na,) P2 attention parameter
        - ctrl (ArrayImpl): (nu,) control input
        - subgame (Tuple): subgame information
        """

        def phi_a(z):
            return (softmax(z, 0) - softmax(z, 1)) * phi_b(z)

        def phi_b(z):
            return softmax(z, 0) * softmax(z, 1)

        (Z_P1, Z_P2, zeta_P1, zeta_P2, x_ph_nom, znom_P1, znom_P2, nom_cost_P1, nom_cost_P2) = subgame

        # State variables.
        z = jnp.hstack((z1, z2))

        # Compute subgame value functions.
        zeta1_11 = zeta_P1[:, 0, 0]
        V1_11 = (x_ph - x_ph_nom[:, 0, 0]).T @ Z_P1[:, :, 0, 0] @ (x_ph - x_ph_nom[:, 0, 0]) + zeta1_11[
            np.newaxis] @ (x_ph - x_ph_nom[:, 0, 0]) + nom_cost_P1[0, 0]
        zeta1_12 = zeta_P1[:, 0, 1]
        V1_12 = (x_ph - x_ph_nom[:, 0, 1]).T @ Z_P1[:, :, 0, 1] @ (x_ph - x_ph_nom[:, 0, 1]) + zeta1_12[
            np.newaxis] @ (x_ph - x_ph_nom[:, 0, 1]) + nom_cost_P1[0, 1]
        zeta1_21 = zeta_P1[:, 1, 0]
        V1_21 = (x_ph - x_ph_nom[:, 1, 0]).T @ Z_P1[:, :, 1, 0] @ (x_ph - x_ph_nom[:, 1, 0]) + zeta1_21[
            np.newaxis] @ (x_ph - x_ph_nom[:, 1, 0]) + nom_cost_P1[1, 0]
        zeta1_22 = zeta_P1[:, 1, 1]
        V1_22 = (x_ph - x_ph_nom[:, 1, 1]).T @ Z_P1[:, :, 1, 1] @ (x_ph - x_ph_nom[:, 1, 1]) + zeta1_22[
            np.newaxis] @ (x_ph - x_ph_nom[:, 1, 1]) + nom_cost_P1[1, 1]

        zeta2_11 = zeta_P2[:, 0, 0]
        V2_11 = (x_ph - x_ph_nom[:, 0, 0]).T @ Z_P2[:, :, 0, 0] @ (x_ph - x_ph_nom[:, 0, 0]) + zeta2_11[
            np.newaxis] @ (x_ph - x_ph_nom[:, 0, 0]) + nom_cost_P2[0, 0]
        zeta2_12 = zeta_P2[:, 0, 1]
        V2_12 = (x_ph - x_ph_nom[:, 0, 1]).T @ Z_P2[:, :, 0, 1] @ (x_ph - x_ph_nom[:, 0, 1]) + zeta2_12[
            np.newaxis] @ (x_ph - x_ph_nom[:, 0, 1]) + nom_cost_P2[0, 1]
        zeta2_21 = zeta_P2[:, 1, 0]
        V2_21 = (x_ph - x_ph_nom[:, 1, 0]).T @ Z_P2[:, :, 1, 0] @ (x_ph - x_ph_nom[:, 1, 0]) + zeta2_21[
            np.newaxis] @ (x_ph - x_ph_nom[:, 1, 0]) + nom_cost_P2[1, 0]
        zeta2_22 = zeta_P2[:, 1, 1]
        V2_22 = (x_ph - x_ph_nom[:, 1, 1]).T @ Z_P2[:, :, 1, 1] @ (x_ph - x_ph_nom[:, 1, 1]) + zeta2_22[
            np.newaxis] @ (x_ph - x_ph_nom[:, 1, 1]) + nom_cost_P2[1, 1]

        # Compute game Hessians.
        a1 = phi_a(z1) * (softmax(z2, 0) * (V1_11-V1_21) + softmax(z2, 1) * (V1_12-V1_22))
        a2 = phi_a(z2) * (softmax(z1, 0) * (V2_11-V2_12) + softmax(z1, 1) * (V2_21-V2_22))
        b1 = phi_b(z1) * phi_b(z2) * (-V1_11 - V1_22 + V1_12 + V1_21)
        b2 = phi_b(z1) * phi_b(z2) * (-V2_11 - V2_22 + V2_12 + V2_21)

        Gamma_1 = horzcat(a1, b1)
        Gamma_2 = horzcat(b2, a2)
        Gamma = vertcat(Gamma_1, Gamma_2)

        # Normalize by subtracting the average.
        Gamma -= (a1+a2+b1+b2) / 4.0

        H_tmp = np.array([[1, -1], [-1, 1]])
        H = kron(Gamma, H_tmp)

        # Compute the opinion state time derivative.
        att_1_vec = att1 * np.ones((self._num_opn_P1,))
        att_2_vec = att2 * np.ones((self._num_opn_P2,))
        att_vec = np.hstack((att_1_vec, att_2_vec))

        D = np.diag(self._damping_opn * np.ones(self._num_opn_P1 + self._num_opn_P2,))

        bias = np.hstack((self._z_P1_bias, self._z_P2_bias))

        z_dot = -D @ z + att_vec * np.tanh(H@z + bias)

        return z + self._T * z_dot


class Car4D(DynamicalSystem):
    """
    4D car model.
    
    Dynamics are as follows:
        \dot x     = v cos theta
        \dot y     = v sin theta
        \dot theta = v * tan(u2) / l
        \dot v     = u1
    """

    def __init__(self, l=3.0, T=0.1):
        """
        Initializer.

        Args:
        - l (float): inter-axle length (m)
        - T (float): time step (s)
        """
        
        self._l = l
        super(Car4D, self).__init__(4, 2, T)

    @partial(jit, static_argnums=(0,))
    def cont_time_dyn(self, x: ArrayImpl, u: ArrayImpl, k: int = 0, *args) -> ArrayImpl:
        """
        Computes the time derivative of state for a particular state/control.

        Args:
        - x (ArrayImpl): (nx,)
        - u (ArrayImpl): (nu,)

        Returns:
        - ArrayImpl: next state (nx,)
        """

        x0_dot = x[3] * jnp.cos(x[2])
        x1_dot = x[3] * jnp.sin(x[2])
        x2_dot = x[3] * jnp.tan(u[1]) / self._l
        x3_dot = u[0]

        return jnp.hstack((x0_dot, x1_dot, x2_dot, x3_dot))


class TwoCar8D(object):
    """
    Two car model for casadi.
    """

    def __init__(self, l=3.0, T=0.1):
        """
        Initializer.

        Args:
        - l (float): inter-axle length (m)
        - T (float): time step (s)
        """

        self._l = l
        self._T = T

    def cont_time_dyn_cas(self, x, u):
        """
        Computes the time derivative of state for a particular state/control.

        Args:
        - x: state vector (8-by-1 opti/MX variable)
        - u: control vector (4-by-1 opti/MX variable)

        Returns:
        - x_dot: time derivative of state (8-by-1 opti/MX variable)
        """

        x0_dot = x[3, 0] * np.cos(x[2, 0])
        x1_dot = x[3, 0] * np.sin(x[2, 0])
        x2_dot = x[3, 0] * np.tan(u[1, 0]) / self._l
        x3_dot = u[0, 0]
        x4_dot = x[7, 0] * np.cos(x[6, 0])
        x5_dot = x[7, 0] * np.sin(x[6, 0])
        x6_dot = x[7, 0] * np.tan(u[3, 0]) / self._l
        x7_dot = u[2, 0]

        return vertcat(x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot)

    def disc_time_dyn_cas(self, x, u):
        """
        Computes the next state in discrete time.

        Args:
        - x: state vector (8-by-1 opti/MX variable)
        - u: control vector (4-by-1 opti/MX variable)

        Returns:
        - x: next state vector (8-by-1 opti/MX variable)
        """

        return x + self._T * self.cont_time_dyn_cas(x, u)

    @partial(jit, static_argnums=(0,))
    def cont_time_dyn_jitted(self, x: ArrayImpl, u: ArrayImpl) -> ArrayImpl:
        """
        Computes the time derivative of state for a particular state/control.

        Args:
        - x: state vector (8-by-1 ArrayImpl)
        - u: control vector (4-by-1 ArrayImpl)

        Returns:
        - x_dot: time derivative of state (8-by-1 ArrayImpl)
        """

        x0_dot = x[3] * jnp.cos(x[2])
        x1_dot = x[3] * jnp.sin(x[2])
        x2_dot = x[3] * jnp.tan(u[1]) / self._l
        x3_dot = u[0]
        x4_dot = x[7] * jnp.cos(x[6])
        x5_dot = x[7] * jnp.sin(x[6])
        x6_dot = x[7] * jnp.tan(u[3]) / self._l
        x7_dot = u[2]

        return jnp.hstack((x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot))

    @partial(jit, static_argnums=(0,))
    def disc_time_dyn_jitted(self, x: ArrayImpl, u: ArrayImpl) -> ArrayImpl:
        """
        Computes the one-step evolution of the system in discrete time with Euler
        integration.

        Args:
        - x: state vector (8-by-1 ArrayImpl)
        - u: control vector (4-by-1 ArrayImpl)

        Returns:
        - x: next state vector (8-by-1 ArrayImpl)
        """

        x_dot = self.cont_time_dyn_jitted(x, u)

        return x + self._T * x_dot
