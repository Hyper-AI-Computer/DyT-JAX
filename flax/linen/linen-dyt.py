import flax.linen as nn
from jax import numpy as jnp

class DyT(nn.Module):
    alpha_init_value: float = 0.5

    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', lambda _: jnp.ones((1)) * self.alpha_init_value)
        weight = self.param('weight', nn.initializers.ones, x.shape)
        bias = self.param('bias', nn.initializers.zeros, x.shape)

        x = jnp.tanh(alpha * x)
        return x * weight + bias
