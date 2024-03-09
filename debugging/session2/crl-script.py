prefix = "crl"

x0 = x
torch.save(x0, f"debugging/session2/tmp/{prefix}-x0.pt")
l0 = self.features_extractor[0]
torch.save(l0, f"debugging/session2/tmp/{prefix}-l0.pt")
s0pre = l0.state_dict()
torch.save(s0pre, f"debugging/session2/tmp/{prefix}-s0pre.pt")
x1 = l0(x0)
s0post = l0.state_dict()
torch.save(s0post, f"debugging/session2/tmp/{prefix}-s0post.pt")

torch.save(x1, f"debugging/session2/tmp/{prefix}-x1.pt")

l1 = self.features_extractor[1]
torch.save(l1, f"debugging/session2/tmp/{prefix}-l1.pt")
s1pre = l1.state_dict()
torch.save(s1pre, f"debugging/session2/tmp/{prefix}-s1pre.pt")
x2 = l1(x1)
s1post = l1.state_dict()
torch.save(s1post, f"debugging/session2/tmp/{prefix}-s1post.pt")

torch.save(x2, f"debugging/session2/tmp/{prefix}-x2.pt")


#########################################

sb3_x0 = torch.load("debugging/session2/tmp/sb3-x0.pt")
sb3_x1 = torch.load("debugging/session2/tmp/sb3-x1.pt")

sb3_s0pre = torch.load("debugging/session2/tmp/sb3-s0pre.pt")
sb3_s0post = torch.load("debugging/session2/tmp/sb3-s0post.pt")
[torch.equal(list(sb3_s0post.values())[i], list(sb3_s0pre.values())[i]) for i in range(len(list(sb3_s0post.values())))]
# [True, True]

sb3_s1pre = torch.load("debugging/session2/tmp/sb3-s1pre.pt")
sb3_s1post = torch.load("debugging/session2/tmp/sb3-s1post.pt")
[torch.equal(list(sb3_s1post.values())[i], list(sb3_s1pre.values())[i]) for i in range(len(list(sb3_s1post.values())))]
# [True, True, False, False, False]

