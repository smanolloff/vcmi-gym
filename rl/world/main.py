import torch

from ..t10n.typesplit_int_sep.main import TransitionModel
from ..p10n.action.classic_int.main import ActionPredictionModel

from ..constants_v10 import (
    HEX_ATTR_MAP,
    GLOBAL_ATTR_MAP,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
)

class WorldModel:
    def __init__(
        self,
        device=torch.device("cpu"),
        transition_model_file="data/t10n/tonlarwl-model.pt",
        action_prediction_model_file="data/t10n/rcqtcsno-model.pt",
    ):
        def load_weights(model, file):
            model.load_state_dict(torch.load(file, weights_only=True, map_location=device), strict=True)

        self.transition_model = TransitionModel(device)
        self.action_prediction_model = ActionPredictionModel(device)

        load_weights(self.transition_model, transition_model_file)
        load_weights(self.action_prediction_model, action_prediction_model_file)

    def full_transition(state, action):
        while state


if __name__ == "__main__":
    main()
