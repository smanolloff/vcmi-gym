import torch

# crl_obs1 = torch.load("debugging/tmp/crl-obs1.pt")
# sb3_obs1 = torch.load("debugging/tmp/sb3-obs1.pt")

# print("torch.equal(crl_obs1, sb3_obs1): %s" % torch.equal(crl_obs1, sb3_obs1))

crl_masks1 = torch.load("debugging/tmp/crl-masks1.pt")
sb3_masks1 = torch.load("debugging/tmp/sb3-masks1.pt")

print("torch.equal(crl_masks1, sb3_masks1): %s" % torch.equal(crl_masks1, sb3_masks1))
