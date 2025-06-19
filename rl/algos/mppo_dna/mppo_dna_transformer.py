# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# This file contains a modified version of CleanRL's PPO-DNA implementation:
# https://github.com/vwxyzjn/cleanrl/blob/caabea4c5b856f429baa2af8bc973d4994d4c330/cleanrl/ppo_dna_atari_envpool.py
import math
import torch
import torch.nn as nn

from .mppo_dna import debug_args, main, NetworkArgs, AgentNN


class TAgentNN(AgentNN):
    def encode(self, x):
        other, hexes = torch.split(x, [self.dim_other, self.dim_hexes], dim=1)
        z_other = self.encoder_other(other)
        z_hexes = self.encoder_hexes(hexes).mean(dim=1)
        merged = torch.cat((z_other, z_hexes), dim=1)
        return self.encoder_merged(merged)


if __name__ == "__main__":
    kur = debug_args()
    kur.network = NetworkArgs(**{
        "encoder_other": [
            # => (B, 26)
            {"t": "LazyLinear", "out_features": 64},
            {"t": "LeakyReLU"},
            # => (B, 64)
        ],
        "encoder_hexes": [
            # => (B, 165*H)
            dict(t="Unflatten", dim=1, unflattened_size=[165, 170]),
            # => (B, 165, H)

            {"t": "LazyLinear", "out_features": 512},
            # => (B, 165, 512)

            {"t": "HexConvResBlock", "channels": 512, "depth": 3},
            # => (B, 165, 512)

            {"t": "TransformerEncoder", "num_layers": 3, "encoder_layer": {
                "t": "TransformerEncoderLayer",
                "d_model": 512,
                "nhead": 8,
                "dropout": 0.2,
                "batch_first": True
            }},
            # => (B, 165, 512)

            {"t": "Mean", "dim": 1},
            # => (B, 512)
        ],
        "encoder_merged": [
            {"t": "LazyLinear", "out_features": 1024},
            {"t": "LeakyReLU"},
            # => (B, 1024)
        ],
        "actor": {"t": "LazyLinear", "out_features": 2312},
        "critic": {"t": "LazyLinear", "out_features": 1}
    })
    main(kur)
