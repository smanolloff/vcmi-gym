# import numpy as np
from sb3_contrib import MaskablePPO

# XXX: maybe import VcmiEnv and load offset from there?
ACTION_OFFSET = 1


class MPPO_AI:
    def __init__(self, file):
        self.model = MaskablePPO.load(file)

    def predict(self, obs, actmasks):
        action, _states = self.model.predict(obs, action_masks=actmasks[ACTION_OFFSET:])
        return action + ACTION_OFFSET
