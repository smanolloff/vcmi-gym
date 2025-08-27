# USAGE:
# 1. Uncomment the DEBUG return in ExecuTorchModel's predict method
#   (in rl/algos/mppo_dna_heads/mppo_dna_heads_new.py)
# 2. Uncomment the 

import torch
import torch.nn.functional as F
import numpy as np
import json
import os

MODEL_PATH = "/Users/simo/Projects/vcmi-play/Mods/MMAI/models/agicelt2-1756110154-multiout5.pte"

OBS_PATH = "%s/obs.np" % os.path.dirname(__file__)
CPPOUT_PATH = "%s/cppout.json" % os.path.dirname(__file__)

# dummy_input = torch.ones([28114])
dummy_input = torch.as_tensor(np.fromfile(OBS_PATH, dtype=np.float32))
cppout = json.load(open(CPPOUT_PATH, "r"))

from executorch.runtime import Runtime, Verification
rt = Runtime.get()
prog = rt.load_program(MODEL_PATH, verification=Verification.Minimal)
output = prog.load_method("predict").execute((dummy_input,))

print("Input shape: %s, stride: %s, dtype: %d" % (tuple(dummy_input.shape), dummy_input.stride(), dummy_input.dtype))

print("action_logits output[0]: %s" % str(output[0]))
print("hex1_logits   output[1]: %s" % str(output[1]))
print("hex2_logits   output[2]: %s" % str(output[2]))
print("probs_act0    output[3]: %s" % str(output[3]))
print("probs_hex1    output[4]: %s" % str(output[4]))
print("probs_hex2    output[5]: %s" % str(output[5]))
print("mask_action   output[6]: %s" % str(output[6]))
print("mask_hex1     output[7]: %s" % str(output[7]))
print("mask_hex2     output[8]: %s" % str(output[8]))
print("act0          output[9]: %s" % str(output[9]))
print("hex1          output[10]: %s" % str(output[10]))
print("hex2          output[11]: %s" % str(output[11]))
print("action        output[12]: %s" % str(output[12]))
print("obs[...]      output[13]: %s" % str(output[13]))
print("mask[...]     output[14]: %s" % str(output[14]))

print("=========================================")
print("action_logits loss[0]: %s" % F.mse_loss(torch.as_tensor(output[0]), torch.tensor(cppout["t0"])))
print("hex1_logits   loss[1]: %s" % F.mse_loss(torch.as_tensor(output[1]), torch.tensor(cppout["t1"])))
print("hex2_logits   loss[2]: %s" % F.mse_loss(torch.as_tensor(output[2]), torch.tensor(cppout["t2"])))
print("probs_act0    loss[3]: %s" % F.mse_loss(torch.as_tensor(output[3]), torch.tensor(cppout["t3"])))
print("probs_hex1    loss[4]: %s" % F.mse_loss(torch.as_tensor(output[4]), torch.tensor(cppout["t4"])))
print("probs_hex2    loss[5]: %s" % F.mse_loss(torch.as_tensor(output[5]), torch.tensor(cppout["t5"])))
print("mask_action   loss[6]: %s" % F.mse_loss(torch.as_tensor(output[6]).float(), torch.tensor(cppout["t6"]).float()))
print("mask_hex1     loss[7]: %s" % F.mse_loss(torch.as_tensor(output[7]).float(), torch.tensor(cppout["t7"]).float()))
print("mask_hex2     loss[8]: %s" % F.mse_loss(torch.as_tensor(output[8]).float(), torch.tensor(cppout["t8"]).float()))
print("act0          loss[9]: %s" % F.mse_loss(torch.as_tensor(output[9]).float(), torch.tensor(cppout["t9"]).float()))
print("hex1          loss[10]: %s" % F.mse_loss(torch.as_tensor(output[10]).float(), torch.tensor(cppout["t10"]).float()))
print("hex2          loss[11]: %s" % F.mse_loss(torch.as_tensor(output[11]).float(), torch.tensor(cppout["t11"]).float()))
print("action        loss[12]: %s" % F.mse_loss(torch.as_tensor(output[12]).float(), torch.tensor(cppout["t12"]).float()))
