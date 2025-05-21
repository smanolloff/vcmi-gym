import torch
from torch import nn

from .t10n import t10n
from .p10n import p10n

from .util.misc import layer_init
from .util.constants_v12 import (
    GLOBAL_ATTR_MAP,
    N_ACTIONS,
)

IDX_BSAP_START = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][1]
IDX_BSAP_END = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][2] + IDX_BSAP_START

IDX_WINNER_START = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1]
IDX_WINNER_END = GLOBAL_ATTR_MAP["BATTLE_WINNER"][2] + IDX_WINNER_START

# Attacker army: 1 phoenix
# Defender army: 3 arrow towers + 8 stacks (incl. ballista)
# NOTE: tent, catapult & arrow towers are excluded from the transitions
# => Transitions:
#   1. Phoenix
#   ---- transitions start:
#   4. 8x stacks (wait)
#   5. 8x stacks (act)
#   ---- transitions end
#   = 16... (the "prediction" will likely be totally wrong)
MAX_TRANSITIONS = 17


class ImaginationCoreModel:
    def __init__(
        self,
        transition_model_file="data/world/t10n/cvftmtsn-model.pt",
        action_prediction_model_file="data/world/p10n/czzklpfu-model.pt",
        device=torch.device("cpu"),
    ):
        def load_weights(model, file):
            model.load_state_dict(torch.load(file, weights_only=True, map_location=device), strict=True)

        self.transition_model = t10n.TransitionModel(device)
        self.transition_model.eval()

        self.action_prediction_model = p10n.ActionPredictionModel(device)
        self.action_prediction_model.eval()

        load_weights(self.transition_model, transition_model_file)
        load_weights(self.action_prediction_model, action_prediction_model_file)

    def forward(self, state, action, t10n_strategy, p10n_strategy, callback=None):
        initial_player = state[:, IDX_BSAP_START:IDX_BSAP_END].argmax(dim=1)
        initial_winner = state[:, IDX_WINNER_START:IDX_WINNER_END].argmax(dim=1)
        # => (B)  # values 0=done, 1=red or 2=blue

        if callback:
            callback(state[0].numpy(), action[0].item())

        # We need the logits for rendering later when strategy=PROBS
        state_logits = self.transition_model(state, action)
        state = self.transition_model.reconstruct(state_logits, strategy=t10n_strategy)

        for _ in range(MAX_TRANSITIONS):
            current_player = state[:, IDX_BSAP_START:IDX_BSAP_END].argmax(dim=1)
            current_winner = state[:, IDX_WINNER_START:IDX_WINNER_END].argmax(dim=1)

            # More transitions are needed as long as the other player is active
            idx_in_progress = torch.nonzero(
                (current_player != initial_player) & (current_winner == initial_winner),
                as_tuple=True
            )[0]

            if idx_in_progress.numel() == 0:
                break

            state_in_progress = state[idx_in_progress]
            state_logits_in_progress = state_logits[idx_in_progress]

            if p10n_strategy == p10n.Prediction.PROBS:
                action_probs_in_progress = self.action_prediction_model.predict_(state_in_progress, strategy=p10n_strategy)
                action_in_progress = action_probs_in_progress.argmax(dim=1)
            else:
                action_in_progress = self.action_prediction_model.predict_(state_in_progress, strategy=p10n_strategy)

            # p10n predicts -1 when it believes battle has ended
            # => some of the "in_progress" states will have a reset action
            # (must filter them out)
            idx_in_progress_valid_action = (action_in_progress != -1).nonzero(as_tuple=True)[0]
            # ^ indexes of `idx_in_progress`
            # e.g. if B=10
            # and idx_in_progress = [2, 3]              // means B=2 and B=3 are in progress
            # and idx_in_progress_valid_action = [0]    // means B=2 has valid action

            idx_in_progress = idx_in_progress[idx_in_progress_valid_action]
            state_in_progress = state_in_progress[idx_in_progress_valid_action]
            state_logits_in_progress = state_logits_in_progress[idx_in_progress_valid_action]
            action_in_progress = action_in_progress[idx_in_progress_valid_action]

            # Transition to next state:
            if p10n_strategy == p10n.Prediction.PROBS:
                action_probs_in_progress = action_probs_in_progress[idx_in_progress_valid_action]
                state_logits[idx_in_progress] = self.transition_model.forward_probs(state_in_progress, action_probs_in_progress)
            else:
                state_logits[idx_in_progress] = self.transition_model(state_in_progress, action_in_progress)

            state[idx_in_progress] = self.transition_model.reconstruct(state_logits[idx_in_progress], strategy=t10n_strategy)

            if callback:
                if t10n_strategy == t10n.Reconstruction.PROBS:
                    # Rendering probs will likely fail => collapse first
                    greedy = self.transition_model.reconstruct(state_logits_in_progress, strategy=t10n.Reconstruction.GREEDY)
                    callback(greedy[0].numpy(), action_in_progress.item())
                else:
                    # Rendering greedy is ok, samples is kind-of-ok => leave as-is
                    callback(state_in_progress[0].numpy(), action_in_progress.item())

        if idx_in_progress.numel() > 0:
            print(f"WARNING: state still in progress after {MAX_TRANSITIONS} transitions")

        res = state

        if t10n_strategy == t10n.Reconstruction.PROBS:
            res = self.transition_model.reconstruct(state_logits, strategy=t10n.Reconstruction.GREEDY)

        if callback:
            # Latest state should have no action
            callback(res[0].numpy(), -1)

        return res


class ImaginationRolloutModel(nn.Module):
    def __init__(
        self,
        transition_model_file,
        action_prediction_model_file,
        horizon=5,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.imagination_core_model = ImaginationCoreModel(transition_model_file, action_prediction_model_file)
        self.horizon = horizon

    def forward(self, state, action, t10n_strategy, p10n_strategy, callback=None):
        for step in range(self.horizon):
            next_state = self.imagination_core_model(state, action, t10n_strategy, p10n_strategy, callback)


class ImaginationAggregatorModel(nn.Module):
    def __init__(
        self,
        transition_model_file,
        action_prediction_model_file,
        num_trajectories=10,  # valid actions are ~65 on average; empirical max is 175
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.imagination_rollout_model = ImaginationRolloutModel(transition_model_file, action_prediction_model_file)
        self.horizon = horizon
        self.num_trajectories = num_trajectories

    def forward(self, obs, mask):
        # Sample N=num_trajectories indices per batch element:
        B, A = mask.shape
        # 1) Count valid actions per row
        valid_counts = mask.sum(dim=1, keepdim=True) # => (B, 1)
        # 2) Build a uniform probability distribution over valid actions
        probs = mask.float() / valid_counts  # => (B, A)
        # 3) replacement=False if N <= min(valid_counts), else True
        replacement = (valid_counts.min().item() < self.num_trajectories)
        actions = torch.multinomial(probs, self.num_trajectories, replacement=replacement)
        # => (B, N)

        n_rollouts = self.imagination_rollout_model(obs, actions)
        
        # TODO: how to aggregate N rollouts?
        # Paper says concat, GPT says maxpool and other approaches
        # return num_trajectories???


class ModelFreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.LazyLinear(16),
            nn.LeakyReLU(),
            nn.LazyLinear(16),
            nn.LeakyReLU(),
        )

        self.head = nn.LazyLinear(32)

    def forward(self, obs):
        z = self.body(obs)
        return self.head(z)


class I2AModel(nn.Module):
    def __init__(
        self,
        transition_model_file,
        action_prediction_model_file,
        horizon,
        device=torch.device("cpu"),
    ):
        self.model_free_model = ModelFreeModel()
        self.imagination_aggregator_model = ImaginationAggregatorModel()

        self.body = nn.Sequential(
            nn.LazyLinear(16),
            nn.LeakyReLU()
        )

        self.action_head = nn.LazyLinear(N_ACTIONS)
        self.value_head = nn.LazyLinear(1)
        layer_init(self)

    def forward(self, obs, action_mask):
        c_mf = self.model_free_model(obs, action_mask)
        c_ia = self.imagination_aggregator_model(obs)
        c = torch.cat((c_ia, c_mf), dim=1)
        action_logits = self.action_head(c)
        value = self.value_head(c)
        return action_logits, value
