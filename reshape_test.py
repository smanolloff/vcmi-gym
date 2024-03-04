B = 2  # batch dim
N = 3  # 3 attributes per hex
obs = torch.as_tensor(np.ndarray((B,1,11,15*N), dtype="int"))
for b in range(B):
    for y in range(11):
        for x in range(15):
            for a in range(N):
                if b == 0:
                    obs[b][0][y][x*N+a] = 100 * (y*15 + x) + a
                else:
                    obs[b][0][y][x*N+a] = -100 * (y*15 + x) - a

v = VcmiHexAttrsAsChannels(3,11,15)
v(obs)


def t1(obs):
    obs.reshape([2,1,11,15,3]).permute(0,4,2,3,1).flatten(start_dim=-2)

def t2(obs):
    tmp = reshape_fortran(obs.flatten(), [3, 2 * 165]).reshape(2 * 3, 165)
    reshape_fortran(tmp, [2, 3, 165]).reshape(2, 3, 11, 15)
