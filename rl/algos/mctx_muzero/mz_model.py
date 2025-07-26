import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
import haiku as hk

from rl.world.util.constants_v12 import (
    N_ACTIONS,
    DIM_OBS,
    DIM_OTHER,
    STATE_SIZE_ONE_HEX,
)


class HexConv(hk.Module):
    def __init__(self, out_channels: int, name: str = None):
        super().__init__(name=name)

        # Precompute the 7 neighbor offsets for even/odd rows
        offsets0 = np.array([-17, -16, -1, 0, 1, 17, 18], np.int32)
        offsets1 = np.array([-18, -17, -1, 0, 1, 16, 17], np.int32)

        # Build the 11×15×7 index array
        inds = np.zeros((11, 15, 7), np.int32)

        for y in range(1, 12):
            for x in range(1, 16):
                base = y * 17 + x
                off = offsets0 if (y % 2 == 0) else offsets1
                inds[y-1, x-1, :] = off + base

        # Store as a DeviceArray for use in __call__
        self.padded_convinds = jnp.array(inds.flatten(), dtype=jnp.int32)

        # Dense layer to project each 7-neighborhood to out_channels
        self.fc = hk.Linear(output_size=out_channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch, 165, in_features]
        b, _, hexdim = x.shape
        x4 = x.reshape((b, 11, 15, hexdim))

        # Add 1 hex padding on both ends of X and Y (11×15 -> 13×17)
        # padded = jnp.zeros((b, 13, 17, hexdim), dtype=x.dtype)
        # padded = padded.at[:, 1:12, 1:16, :].set(x4)
        # Faster version: (before_1, after_1) values to pad for each axis
        padded = jnp.pad(x4, ((0, 0), (1, 1), (1, 1), (0, 0)))

        # Gather all 7 neighbors per hex and apply dense
        flat = padded.reshape((b, -1, hexdim))
        idxs = self.padded_convinds  # shape (11*15*7,)
        neigh = flat[:, idxs, :].reshape((b, 165, 7 * hexdim))
        return self.fc(neigh)


class HexConvResLayer(hk.Module):
    def __init__(self, channels: int, name: str = None):
        super().__init__(name=name)
        self.conv1 = HexConv(channels)
        self.conv2 = HexConv(channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.conv1(x)
        y = jnn.relu(y)
        y = self.conv2(y)
        return jnn.relu(y + x)


class HexConvResBlock(hk.Module):
    def __init__(self, channels: int, depth: int, name: str = None):
        super().__init__(name=name)
        self.layers = [HexConvResLayer(channels) for _ in range(depth)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


# Main MuZero model with 2 heads: action and value
class MZModel(hk.Module):
    def __init__(self, depth: int = 3, name: str = None):
        super().__init__(name=name)
        self.encoder_other = hk.Sequential([
            hk.Linear(64),
            jnn.relu
        ])

        self.encoder_hexes = hk.Sequential([
            HexConvResBlock(channels=170, depth=depth),
            hk.Linear(32),
            jnn.relu,
        ])

        self.encoder_merged = hk.Sequential([
            hk.Linear(1024),
            jnn.relu
        ])

        self.action_head = hk.Linear(N_ACTIONS)
        self.value_head = hk.Linear(1)

    def __call__(self, obs):
        b = obs.shape[0]
        other, hexes = jnp.split(obs, [DIM_OTHER], axis=1)

        z_other = self.encoder_other(other)
        # (B, Z_OTHER)

        in_hexes = hexes.reshape(b, 165, STATE_SIZE_ONE_HEX)
        z_hexes = self.encoder_hexes(in_hexes)
        # (B, 165, Z_HEX)

        merged = jnp.concat([z_other, z_hexes.reshape(b, -1)], axis=1)
        # (B, Z_OTHER + Z_HEX)

        z_merged = self.encoder_merged(merged)
        # (B, Z_MERGED)

        action_logits = self.action_head(z_merged)
        value = self.value_head(z_merged)

        return action_logits, value


if __name__ == "__main__":
    def forward_fn(obs):
        model = MZModel()
        return model(obs)

    rng = jax.random.PRNGKey(0)
    haiku_fwd = hk.transform(forward_fn)

    # create a jitted apply that compiles once and reuses the XLA binary
    @jax.jit
    def jit_fwd(params, rng, obs):
        return haiku_fwd.apply(params, rng, obs)

    haiku_params = haiku_fwd.init(rng, obs=jnp.zeros([2, DIM_OBS]))
    res1 = haiku_fwd.apply(haiku_params, rng, jnp.zeros([2, DIM_OBS]))
    res2 = jit_fwd(haiku_params, rng, jnp.zeros([2, DIM_OBS]))

    import ipdb; ipdb.set_trace()  # noqa
    pass
