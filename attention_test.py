import torch
# import vcmi_gym

# Query: (165, 512)
# --- hex (512 bytes) ----
# qqqqqqqqqqqqqqqqqqqqqqqq | = Q1 (query for Hex1, 512 floats)
# qqqqqqqqqqqqqqqqqqqqqqqq | = Q2 (query for Hex2, 512 floats)
# qqqqqqqqqqqqqqqqqqqqqqqq |
# qqqqqqqqqqqqqqqqqqqqqqqq |
# ... 165 total
# qqqqqqqqqqqqqqqqqqqqqqqq | = Q165

# Key: (165, 512)
# Key (Transposed): (512, 165)

# k k k k k k k k k k k  |
# k k k k k k k k k k k  |
# k k k k k k k k k k k  |
# k k k k k k k k k k k  |
# k k k k k k k k k k k  |
# k k k k k k k k k k k  |
# k k k k k k k k k k k  |
# ... 512 total
# k k k k k k k k k k k  |
# | | |               |
# K1| |               K165
#   K2|
#     K3


# env = vcmi_gym.VcmiEnv("ai/generated/B001.vmap")
# anet = torch.nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)

# obs = torch.ones(1, 11, 240).reshape(1, 165, 16)
# mask1 = torch.tensor(1, dtype=torch.float32).broadcast_to(165, 165)
# mask2 = torch.tensor(0, dtype=torch.float32).broadcast_to(165, 165)
# mask3 = torch.tensor(True).broadcast_to(165, 165)
# mask4 = torch.tensor(False).broadcast_to(165, 165)
# mask5 = torch.tensor(-1e9, dtype=torch.float32).broadcast_to(165, 165)

# a1 = anet(obs, obs, obs, need_weights=True, attn_mask=mask1)
# a2 = anet(obs, obs, obs, need_weights=True, attn_mask=mask2)
# a3 = anet(obs, obs, obs, need_weights=True, attn_mask=mask3)
# a4 = anet(obs, obs, obs, need_weights=True, attn_mask=mask4)
# a5 = anet(obs, obs, obs, need_weights=True, attn_mask=mask5)


inputs = torch.ones((8, 2, 6))
mha = torch.nn.MultiheadAttention(embed_dim=6, num_heads=2)
attn_mask = torch.tril(torch.ones((8, 8)).bool())
outputs, weights = mha(inputs, inputs, inputs, attn_mask=attn_mask)

print("%s\n%s" % (attn_mask, weights))
# tensor([[ True, False, False, False, False, False, False, False],
#         [ True,  True, False, False, False, False, False, False],
#         [ True,  True,  True, False, False, False, False, False],
#         [ True,  True,  True,  True, False, False, False, False],
#         [ True,  True,  True,  True,  True, False, False, False],
#         [ True,  True,  True,  True,  True,  True, False, False],
#         [ True,  True,  True,  True,  True,  True,  True, False],
#         [ True,  True,  True,  True,  True,  True,  True,  True]])
# tensor([[[0.0000, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429],
#          [0.0000, 0.0000, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],
#          [0.0000, 0.0000, 0.0000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.2500, 0.2500, 0.2500],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3333, 0.3333, 0.3333],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.5000],
#          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
#          [   nan,    nan,    nan,    nan,    nan,    nan,    nan,    nan]],


mat1 = torch.as_tensor([
    [1, 2, 3],
    [4, 5, 6],
])

mat2 = torch.as_tensor([
    [7, 10],
    [8, 11],
    [9, 12]
])

# mat2 = torch.transpose(mat2, 0, 1)

torch.matmul(mat1, mat2)
# tensor([[ 50,  68],
#         [122, 167]])
