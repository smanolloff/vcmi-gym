import torch
import torch.nn.functional as F
import numpy as np
import json
import os

MODEL_PATH = "/Users/simo/Projects/vcmi-play/Mods/MMAI/models/cgfpenwh-1756282288-quantized.pte"

OBS_PATH = "%s/obs2.np" % os.path.dirname(__file__)
CPPOUT_PATH = "%s/cppout.json" % os.path.dirname(__file__)

# dummy_input = torch.ones([28114])
dummy_input = torch.as_tensor(np.fromfile(OBS_PATH, dtype=np.float32))
cppout = json.load(open(CPPOUT_PATH, "r"))

# Ensure results from the same observation are compared
assert cppout["obs_path"] == OBS_PATH, f"OBS_PATH mismatch: {cppout['obs_path']} == {OBS_PATH}"

# Don't assert (might want to test different models)
if cppout["model_path"] != MODEL_PATH:
    print(" ****")
    print(f" **** WARNING: MODEL_PATH mismatch:\ncpp: {cppout['model_path']}\npy:  {MODEL_PATH}")
    print(" ****")

from executorch.runtime import Runtime, Verification
rt = Runtime.get()
prog = rt.load_program(MODEL_PATH, verification=Verification.Minimal)
output = prog.load_method("predict").execute((dummy_input,))

print("Input shape: %s, stride: %s, dtype: %s" % (tuple(dummy_input.shape), dummy_input.stride(), dummy_input.dtype))

# print("action_logits output[0]: %s" % str(output[0]))
# print("hex1_logits   output[1]: %s" % str(output[1]))
# print("hex2_logits   output[2]: %s" % str(output[2]))
# print("probs_act0    output[3]: %s" % str(output[3]))
# print("probs_hex1    output[4]: %s" % str(output[4]))
# print("probs_hex2    output[5]: %s" % str(output[5]))
# print("mask_action   output[6]: %s" % str(output[6]))
# print("mask_hex1     output[7]: %s" % str(output[7]))
# print("mask_hex2     output[8]: %s" % str(output[8]))
# print("act0          output[9]: %s" % str(output[9]))
# print("hex1          output[10]: %s" % str(output[10]))
# print("hex2          output[11]: %s" % str(output[11]))
# print("action        output[12]: %s" % str(output[12]))
# print("obs[...]      output[13]: %s" % str(output[13]))
# print("mask[...]     output[14]: %s" % str(output[14]))

print("")
print("=============== LOSSES (cpp vs. py-loaded) ===============")
print("")
print(f"py_model:  {MODEL_PATH}")
print(f"cpp_model: {cppout['model_path']}")
print("")

print("action value (cpp):          %d" % cppout["results"][12]["data"][0])
print("action value (py-loaded):    %d" % output[12][0].item())
print("---")

for i in range(len(cppout["results"])):
    t0 = torch.as_tensor(output[i])
    t1 = torch.tensor(cppout["results"][i]["data"]).float().expand_as(t0)
    print("%-15s: %.4f" % (cppout["results"][i]["name"], F.mse_loss(t0, t1).item()))

print("")
print("=============== LOSSES (cpp vs. py-original) ===============")
print("")

from rl.algos.mppo_dna_heads.mppo_dna_heads_new import DNAModel

filebase = "cgfpenwh-1756282288"
model_cfg_path = f"{filebase}-config.json"
model_weights_path = f"{filebase}-model-dna.pt"

print(f"original_model: {os.path.abspath(model_weights_path)}")
print(f"cpp_model:      {cppout['model_path']}")

print("")
print("NOTE: The origial model is non-deterministic!")
print("      Repeated invocations may yield different results.")
print("      Most often, though, the losses should be 0")
print("      (those samples should have highest probability)")
print("")

config = json.load(open(model_cfg_path, "r"))["model"]
model = DNAModel(config, device=torch.device("cpu"))
weights = torch.load(model_weights_path, weights_only=True, map_location="cpu")
model.load_state_dict(weights)
with torch.no_grad():
    actdata = model.model_policy.get_actdata_eval(dummy_input.unsqueeze(0))

# Cannot compare logits (the dist outputs normalized logits)
# print("action_logits : %.3f" % F.mse_loss(actdata.act0_dist.logits.squeeze(0), cppout["results"][0]["data"]))
# print("hex1_logits   : %.3f" % F.mse_loss(actdata.hex1_dist.logits.squeeze(0), cppout["results"][1]["data"]))
# print("hex2_logits   : %.3f" % F.mse_loss(actdata.hex2_dist.logits.squeeze(0), cppout["results"][2]["data"]))
print("action value (cpp):          %d" % cppout["results"][12]["data"][0])
print("action value (py-original):  %d" % actdata.action[0])
print("---")
print("probs_act0    : %.4f" % F.mse_loss(actdata.act0_dist.probs.squeeze(0), torch.tensor(cppout["results"][3]["data"]).float()).item())
print("probs_hex1    : %.4f" % F.mse_loss(actdata.hex1_dist.probs.squeeze(0), torch.tensor(cppout["results"][4]["data"]).float()).item())
print("probs_hex2    : %.4f" % F.mse_loss(actdata.hex2_dist.probs.squeeze(0), torch.tensor(cppout["results"][5]["data"]).float()).item())
print("mask_action   : %.4f" % F.mse_loss(actdata.act0_dist.mask.squeeze(0), torch.tensor(cppout["results"][6]["data"]).float()).item())
print("mask_hex1     : %.4f" % F.mse_loss(actdata.hex1_dist.mask.squeeze(0), torch.tensor(cppout["results"][7]["data"]).float()).item())
print("mask_hex2     : %.4f" % F.mse_loss(actdata.hex2_dist.mask.squeeze(0), torch.tensor(cppout["results"][8]["data"]).float()).item())
print("act0          : %.4f" % F.mse_loss(actdata.act0, torch.tensor(cppout["results"][9]["data"]).float()).item())
print("hex1          : %.4f" % F.mse_loss(actdata.hex1, torch.tensor(cppout["results"][10]["data"]).float()).item())
print("hex2          : %.4f" % F.mse_loss(actdata.hex2, torch.tensor(cppout["results"][11]["data"]).float()).item())
print("action        : %.4f" % F.mse_loss(actdata.action, torch.tensor(cppout["results"][12]["data"]).float()).item())
