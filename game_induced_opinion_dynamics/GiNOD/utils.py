"""
utils.py

This file contains utility functions for the Game-induced Nonlinear
Opinion Dynamics (GiNOD) project.
"""

from functools import partial
from jax import jit
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp


@partial(jit, static_argnums=(1,))
def softmax(z: ArrayImpl, idx: int) -> ArrayImpl:
    """
    Softmax operator.

    Args:
    - z (ArrayImpl): vector
    - idx (int): index

    Returns:
    - ArrayImpl: output
    """
    
    return jnp.exp(z[idx]) / jnp.sum(jnp.exp(z))
