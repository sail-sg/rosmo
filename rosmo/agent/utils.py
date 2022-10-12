# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent utilities."""
# Codes adapted from:
# https://github.com/Hwhitetooth/jax_muzero/blob/main/algorithms/utils.py
import chex
import jax
import jax.numpy as jnp

from rosmo.type import Array


def scale_gradient(g: Array, scale: float) -> Array:
    """Scale the gradient.

    Args:
        g (_type_): Parameters that contain gradients.
        scale (float): Scale.

    Returns:
        Array: Parameters with scaled gradients.
    """
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def scalar_to_two_hot(x: Array, num_bins: int) -> Array:
    """A categorical representation of real values.

    Ref: https://www.nature.com/articles/s41586-020-03051-4.pdf.

    Args:
        x (Array): Scalar data.
        num_bins (int): Number of bins.

    Returns:
        Array: Distributional data.
    """
    max_val = (num_bins - 1) // 2
    x = jnp.clip(x, -max_val, max_val)
    x_low = jnp.floor(x).astype(jnp.int32)
    x_high = jnp.ceil(x).astype(jnp.int32)
    p_high = x - x_low
    p_low = 1.0 - p_high
    idx_low = x_low + max_val
    idx_high = x_high + max_val
    cat_low = jax.nn.one_hot(idx_low, num_bins) * p_low[..., None]
    cat_high = jax.nn.one_hot(idx_high, num_bins) * p_high[..., None]
    return cat_low + cat_high


def logits_to_scalar(logits: Array, num_bins: int) -> Array:
    """The inverse of the scalar_to_two_hot function above.

    Args:
        logits (Array): Distributional logits.
        num_bins (int): Number of bins.

    Returns:
        Array: Scalar data.
    """
    chex.assert_equal(num_bins, logits.shape[-1])
    max_val = (num_bins - 1) // 2
    x = jnp.sum((jnp.arange(num_bins) - max_val) * jax.nn.softmax(logits), axis=-1)
    return x


def value_transform(x: Array, epsilon: float = 1e-3) -> Array:
    """A non-linear value transformation for variance reduction.

    Ref: https://arxiv.org/abs/1805.11593.

    Args:
        x (Array): Data.
        epsilon (float, optional): Epsilon. Defaults to 1e-3.

    Returns:
        Array: Transformed data.
    """
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + epsilon * x


def inv_value_transform(x: Array, epsilon: float = 1e-3) -> Array:
    """The inverse of the non-linear value transformation above.

    Args:
        x (Array): Data.
        epsilon (float, optional): Epsilon. Defaults to 1e-3.

    Returns:
        Array: Inversely transformed data.
    """
    return jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * epsilon * (jnp.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon))
        ** 2
        - 1
    )
