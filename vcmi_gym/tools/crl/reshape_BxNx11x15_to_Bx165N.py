import torch
import numpy as np

#
# Test code for Bx1x11x15N_to_Bx165N
#

B = 2  # batch dim
N = 3  # 3 attributes per hex
obs = torch.as_tensor(np.ndarray((B, N, 11, 15), dtype="int"))
for b in range(B):
    for y in range(11):
        for x in range(15):
            for a in range(N):
                if b == 0:
                    obs[b][a][y][x] = 100 * (y*15 + x) + a
                else:
                    obs[b][a][y][x] = -100 * (y*15 + x) - a

obs.permute(0, 2, 3, 1).flatten(start_dim=1)
