import typing
from flax import nnx
from flax.nnx.nn import initializers
from jax import numpy as jnp
from flax.typing import Dtype, Initializer

class DyT(nnx.Module):
    def __init__(
            self,
            num_features,
            *,
            alpha_init_value: float = 0.5,
            weight_init: Initializer = initializers.ones_init(),
            bias_init: Initializer = initializers.zeros_init(),
            dtype: typing.Optional[Dtype] = None,
            rngs: nnx.Rngs,
        ):
        self.alpha = nnx.Param(initializers.ones_init()(rngs.params(), (1,), dtype) * alpha_init_value)
        self.weight = nnx.Param(weight_init(rngs.params(), (num_features,), dtype))
        self.bias = nnx.Param(bias_init(rngs.params(), (num_features,), dtype))

    def __call__(self, x):
        x = jnp.tanh(self.alpha * x)
        x = x * self.weight + self.bias
        return x