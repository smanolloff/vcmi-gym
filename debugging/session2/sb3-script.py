# flake8: noqa
#########################################
# # SAVING

# torch = th
# prefix = "sb3"

# x0 = obs
# torch.save(x0, f"debugging/session2/tmp/{prefix}-x0.pt")
# l0 = self.features_extractor.network[0]
# torch.save(l0, f"debugging/session2/tmp/{prefix}-l0.pt")
# s0pre = l0.state_dict()
# torch.save(s0pre, f"debugging/session2/tmp/{prefix}-s0pre.pt")
# x1 = l0(x0)
# s0post = l0.state_dict()
# torch.save(s0post, f"debugging/session2/tmp/{prefix}-s0post.pt")

# torch.save(x1, f"debugging/session2/tmp/{prefix}-x1.pt")

# l1 = self.features_extractor.network[1]
# torch.save(l1, f"debugging/session2/tmp/{prefix}-l1.pt")
# s1pre = l1.state_dict()
# torch.save(s1pre, f"debugging/session2/tmp/{prefix}-s1pre.pt")
# x2 = l1(x1)
# s1post = l1.state_dict()
# torch.save(s1post, f"debugging/session2/tmp/{prefix}-s1post.pt")

# torch.save(x2, f"debugging/session2/tmp/{prefix}-x2.pt")

# #########################################
# # LOADING & COMPARING

# import torch

# # CRL

# crl_x0 = torch.load("debugging/session2/tmp/crl-x0.pt")
# crl_x1 = torch.load("debugging/session2/tmp/crl-x1.pt")

# crl_s0pre = torch.load("debugging/session2/tmp/crl-s0pre.pt")
# crl_s0post = torch.load("debugging/session2/tmp/crl-s0post.pt")
# [torch.equal(list(crl_s0post.values())[i], list(crl_s0pre.values())[i]) for i in range(len(list(crl_s0post.values())))]
# # [True, True]

# crl_s1pre = torch.load("debugging/session2/tmp/crl-s1pre.pt")
# crl_s1post = torch.load("debugging/session2/tmp/crl-s1post.pt")
# [torch.equal(list(crl_s1post.values())[i], list(crl_s1pre.values())[i]) for i in range(len(list(crl_s1post.values())))]
# # [True, True, False, False, False]
# # !!!!!!!!!!!!!

# # SB3

# sb3_x0 = torch.load("debugging/session2/tmp/sb3-x0.pt")
# sb3_x1 = torch.load("debugging/session2/tmp/sb3-x1.pt")

# sb3_s0pre = torch.load("debugging/session2/tmp/sb3-s0pre.pt")
# sb3_s0post = torch.load("debugging/session2/tmp/sb3-s0post.pt")
# [torch.equal(list(sb3_s0post.values())[i], list(sb3_s0pre.values())[i]) for i in range(len(list(sb3_s0post.values())))]
# # [True, True]

# sb3_s1pre = torch.load("debugging/session2/tmp/sb3-s1pre.pt")
# sb3_s1post = torch.load("debugging/session2/tmp/sb3-s1post.pt")
# [torch.equal(list(sb3_s1post.values())[i], list(sb3_s1pre.values())[i]) for i in range(len(list(sb3_s1post.values())))]
# # [True, True, True, True, True]
# # !!!!!!!!!!!!!

# # CRL <> SB3
# torch.equal(sb3_x0, crl_x0)
# # True
# torch.equal(sb3_x1, crl_x1)
# # True


####################
#### FURTHER TESTING FOR L1 (BatchNorm) DIFF
import torch

sb3_l1 = torch.load("debugging/session2/tmp/sb3-l1.pt")
crl_l1 = torch.load("debugging/session2/tmp/crl-l1.pt")

# testing state dicts
[torch.equal(list(sb3_l1.state_dict().values())[i], list(crl_l1.state_dict().values())[i]) for i in range(len(list(sb3_l1.state_dict().values())))]
# [True, True, True, True, True]

# testing output
x1 = torch.load("debugging/session2/tmp/crl-x1.pt")

t1 = torch.ones(x1.shape)
t2 = torch.ones(x1.shape)

sb3_l1.state_dict()["running_mean"]
# tensor([ 0.0218, -0.0098,  0.0457,  0.0313,  0.0205,  0.0288,  0.0430, -0.0093,
#         -0.0079,  0.0109, -0.0617, -0.0137, -0.0126, -0.0342,  0.0377, -0.0526,
#         -0.0642, -0.0031, -0.0101,  0.0094,  0.0237,  0.0205,  0.0050,  0.0195,
#         -0.0459,  0.0266, -0.0410, -0.0609,  0.0482, -0.0077, -0.0110, -0.0300])

sb3_res = sb3_l1(t1)
sb3_l1.state_dict()["running_mean"]
# tensor([ 0.0218, -0.0098,  0.0457,  0.0313,  0.0205,  0.0288,  0.0430, -0.0093,
#         -0.0079,  0.0109, -0.0617, -0.0137, -0.0126, -0.0342,  0.0377, -0.0526,
#         -0.0642, -0.0031, -0.0101,  0.0094,  0.0237,  0.0205,  0.0050,  0.0195,
#         -0.0459,  0.0266, -0.0410, -0.0609,  0.0482, -0.0077, -0.0110, -0.0300])
# ^^^^ SAME

crl_l1.state_dict()["running_mean"]
# tensor([ 0.0218, -0.0098,  0.0457,  0.0313,  0.0205,  0.0288,  0.0430, -0.0093,
#         -0.0079,  0.0109, -0.0617, -0.0137, -0.0126, -0.0342,  0.0377, -0.0526,
#         -0.0642, -0.0031, -0.0101,  0.0094,  0.0237,  0.0205,  0.0050,  0.0195,
#         -0.0459,  0.0266, -0.0410, -0.0609,  0.0482, -0.0077, -0.0110, -0.0300])
crl_res = crl_l1(t1)
crl_l1.state_dict()["running_mean"]
# tensor([0.1197, 0.0912, 0.1412, 0.1281, 0.1185, 0.1259, 0.1387, 0.0917, 0.0929,
#         0.1098, 0.0445, 0.0877, 0.0887, 0.0692, 0.1339, 0.0526, 0.0422, 0.0972,
#         0.0909, 0.1084, 0.1213, 0.1184, 0.1045, 0.1176, 0.0587, 0.1239, 0.0631,
#         0.0452, 0.1434, 0.0930, 0.0901, 0.0730])
# ^^^^^ DIFFERENT

# THE REASON:
sb3_l1.training
# False
crl_l1.training
# True
