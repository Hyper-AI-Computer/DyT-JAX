import flax.linen as nn
from jax import numpy as jnp

class DyT(nn.Module):
    alpha_init_value: float = 0.5
    # https://github.com/Hyper-AI-Computer/DyT-JAX

    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', lambda _: jnp.ones((1)) * self.alpha_init_value)
        weight = self.param('weight', nn.initializers.ones, (x.shape[-1],))
        bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],))

        x = jnp.tanh(alpha * x)
        return x * weight + bias
