import jax.nn as jnn
import jax.numpy as jnp
import haiku as hk

from rl.world.constants_v12 import (
    N_ACTIONS,
    DIM_OTHER,
    STATE_SIZE,
    STATE_SIZE_ONE_HEX,
)


class HexConv(hk.Module):
    def __init__(self, out_channels: int, name: str = None):
        super().__init__(name=name)
        self.out_channels = out_channels

    def setup(self):
        # Precompute the 7 neighbor offsets for even/odd rows
        offsets0 = jnp.array([-17, -16, -1, 0, 1, 17, 18], dtype=jnp.int32)
        offsets1 = jnp.array([-18, -17, -1, 0, 1, 16, 17], dtype=jnp.int32)

        # Build the 11×15×7 index array
        inds = jnp.zeros((11, 15, 7), dtype=jnp.int32)
        for y in range(1, 12):
            for x in range(1, 16):
                base = y * 17 + x
                off = offsets0 if (y % 2 == 0) else offsets1
                inds = inds.at[y-1, x-1].set(off + base)

        # Store as a DeviceArray for use in __call__
        self.padded_convinds = jnp.array(inds.flatten(), dtype=jnp.int32)

        # Dense layer to project each 7-neighborhood to out_channels
        self.fc = hk.Linear(output_size=self.out_channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch, 165, in_features]
        b, _, hexdim = x.shape
        x4 = x.reshape((b, 11, 15, hexdim))

        # Pad by 1 hex on each side (13×17)
        padded = jnp.zeros((b, 13, 17, hexdim), dtype=x.dtype)
        padded = padded.at[:, 1:12, 1:16, :].set(x4)

        # Gather all 7 neighbors per hex and apply dense
        flat = padded.reshape((b, -1, hexdim))
        idxs = self.padded_convinds  # shape (11*15*7,)
        neigh = flat[:, idxs, :].reshape((b, 165, 7 * hexdim))
        return self.fc(neigh)


class HexConvResLayer(hk.Module):
    def __init__(self, channels: int, name: str = None):
        super().__init__(name=name)
        self.channels = channels

    def setup(self):
        self.conv1 = HexConv(self.channels)
        self.conv2 = HexConv(self.channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.conv1(x)
        y = jnn.relu(y)
        y = self.conv2(y)
        return jnn.relu(y + x)


class HexConvResBlock(hk.Module):
    def __init__(self, channels: int, depth: int, name: str = None):
        super().__init__(name=name)
        self.channels = channels
        self.depth = depth

    def setup(self):
        self.layers = [HexConvResLayer(self.channels) for _ in range(self.depth)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


# Trainable prediction head (policy & value)
class MZModel(hk.Module):
    def setup(self):
        self.encoder_other = hk.Sequential([
            hk.Linear(64),
            jnn.relu
        ])

        self.encoder_hexes = hk.Sequential([
            HexConvResBlock(channels=170, layers=3),
            hk.Linear(32),
            jnn.relu,
        ])

        self.encoder_merged = hk.Linear(1024)

        self.action_head = hk.Linear(N_ACTIONS)
        self.value_head = hk.Linear(1)

    def __call__(self, obs):
        other, hexes = jnp.split(obs, [DIM_OTHER], axis=1)

        z_other = self.encoder_other(other)
        # (B, 64)

        z_hexes = self.encoder_hexes(hexes.reshape(-1, 165, STATE_SIZE_ONE_HEX))
        # (B, 165, 32)

        merged = jnp.concat([z_other, z_hexes.reshape(-1, STATE_SIZE)], axis=1)
        # (B, 64 + 5280)

        z_merged = self.encoder_merged(merged)
        # (B, 1024)

        action_logits = self.aciton_head(z_merged)
        value = self.value_head(z_merged)

        return action_logits, value
