import torch
from torch import nn
import torch.nn.functional as F

from .t10n import t10n
from .p10n import p10n_nll as p10n

from .util.misc import layer_init
from .util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    N_ACTIONS,
)

INDEX_BSAP_START = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][1]
INDEX_BSAP_END = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][2] + INDEX_BSAP_START

INDEX_WINNER_START = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1]
INDEX_WINNER_END = GLOBAL_ATTR_MAP["BATTLE_WINNER"][2] + INDEX_WINNER_START

P_OFFSET0 = STATE_SIZE_GLOBAL
P_OFFSET1 = STATE_SIZE_GLOBAL + STATE_SIZE_ONE_PLAYER

INDEX_PLAYER0_DMG_RECEIVED_NOW_REL = P_OFFSET0 + PLAYER_ATTR_MAP["DMG_RECEIVED_NOW_REL"][1]
INDEX_PLAYER1_DMG_RECEIVED_NOW_REL = P_OFFSET1 + PLAYER_ATTR_MAP["DMG_RECEIVED_NOW_REL"][1]
INDEX_PLAYER0_VALUE_LOST_NOW_REL = P_OFFSET0 + PLAYER_ATTR_MAP["VALUE_LOST_NOW_REL"][1]
INDEX_PLAYER1_VALUE_LOST_NOW_REL = P_OFFSET1 + PLAYER_ATTR_MAP["VALUE_LOST_NOW_REL"][1]
INDEX_PLAYER0_VALUE_LOST_ACC_REL0 = P_OFFSET0 + PLAYER_ATTR_MAP["VALUE_LOST_ACC_REL0"][1]
INDEX_PLAYER1_VALUE_LOST_ACC_REL0 = P_OFFSET1 + PLAYER_ATTR_MAP["VALUE_LOST_ACC_REL0"][1]


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


class ImaginationCoreModel(nn.Module):
    def __init__(
        self,
        side,
        reward_step_fixed,
        reward_dmg_mult,
        reward_term_mult,
        transition_model_file="hauzybxn-model.pt",
        action_prediction_model_file="ogyesvkb-model.pt",
        device=torch.device("cpu"),
    ):
        super().__init__()

        def load_weights(model, file):
            model.load_state_dict(torch.load(file, weights_only=True, map_location=device), strict=True)

        self.side = side
        self.reward_step_fixed = torch.tensor(reward_step_fixed, dtype=torch.float32, device=device)
        self.reward_dmg_mult = torch.tensor(reward_dmg_mult, dtype=torch.float32, device=device)
        self.reward_term_mult = torch.tensor(reward_term_mult, dtype=torch.float32, device=device)

        self.transition_model = t10n.TransitionModel(device)
        self.transition_model.eval()

        self.action_prediction_model = p10n.ActionPredictionModel(device)
        self.action_prediction_model.eval()

        load_weights(self.transition_model, transition_model_file)
        load_weights(self.action_prediction_model, action_prediction_model_file)

        assert PLAYER_ATTR_MAP["DMG_RECEIVED_NOW_REL"][2] == 1
        assert PLAYER_ATTR_MAP["VALUE_LOST_NOW_REL"][2] == 1
        assert PLAYER_ATTR_MAP["VALUE_LOST_ACC_REL0"][2] == 1
        assert PLAYER_ATTR_MAP["DMG_RECEIVED_NOW_REL"][3] == 1000
        assert PLAYER_ATTR_MAP["VALUE_LOST_NOW_REL"][3] == 1000
        assert PLAYER_ATTR_MAP["VALUE_LOST_ACC_REL0"][3] == 1000

        if side == 0:
            self.index_my_dmg_received_now_rel = INDEX_PLAYER0_DMG_RECEIVED_NOW_REL
            self.index_my_value_lost_now_rel = INDEX_PLAYER0_VALUE_LOST_NOW_REL
            self.index_my_value_lost_acc_rel0 = INDEX_PLAYER0_VALUE_LOST_ACC_REL0
            self.index_enemy_dmg_received_now_rel = INDEX_PLAYER1_DMG_RECEIVED_NOW_REL
            self.index_enemy_value_lost_now_rel = INDEX_PLAYER1_VALUE_LOST_NOW_REL
            self.index_enemy_value_lost_acc_rel0 = INDEX_PLAYER1_VALUE_LOST_ACC_REL0
        elif side == 1:
            self.index_my_dmg_received_now_rel = INDEX_PLAYER1_DMG_RECEIVED_NOW_REL
            self.index_my_value_lost_now_rel = INDEX_PLAYER1_VALUE_LOST_NOW_REL
            self.index_my_value_lost_acc_rel0 = INDEX_PLAYER1_VALUE_LOST_ACC_REL0
            self.index_enemy_dmg_received_now_rel = INDEX_PLAYER0_DMG_RECEIVED_NOW_REL
            self.index_enemy_value_lost_now_rel = INDEX_PLAYER0_VALUE_LOST_NOW_REL
            self.index_enemy_value_lost_acc_rel0 = INDEX_PLAYER0_VALUE_LOST_ACC_REL0
        else:
            raise Exception("Unknown side: %s" % side)

    def forward(self, initial_state, initial_action, t10n_strategy, p10n_strategy, callback=None, obs0=None):
        initial_player = initial_state[:, INDEX_BSAP_START:INDEX_BSAP_END].argmax(dim=1)
        # => (B)  # values 0=none, 1=red or 2=blue
        initial_winner = initial_state[:, INDEX_WINNER_START:INDEX_WINNER_END].argmax(dim=1)
        # => (B)  # values 0=none, 1=red or 2=blue
        initial_done = (initial_player == 0) | (initial_winner > 0)
        # => (B) of done flags

        if callback:
            callback(initial_state[0].numpy(), initial_action[0].item())

        action = initial_action
        reward = initial_state.new_zeros(initial_state.size(0))
        # => (B) of rewards

        # # DEBUG - use for debugging to simulate 100% accurate prediction
        # debug = {"i": 0}
        # def debugtransition(*args, **kwargs):
        #     debug["i"] += 1
        #     return torch.as_tensor(obs0["transitions"]["observations"][debug["i"]]).unsqueeze(0).clone()
        # # tm = self.transition_model
        # tm = debugtransition

        for i in range(MAX_TRANSITIONS):
            if i == 0:
                current_player = initial_player
                current_winner = initial_winner
                done = initial_done
                state = initial_state
                state_logits = initial_state
                idx_in_progress = torch.nonzero(~done, as_tuple=True)[0]
                # => (B') of indexes
            else:
                current_player = state[:, INDEX_BSAP_START:INDEX_BSAP_END].argmax(dim=1)
                current_winner = state[:, INDEX_WINNER_START:INDEX_WINNER_END].argmax(dim=1)
                done = (current_player == 0) | (current_winner > 0)

                # More transitions are needed as long as the other player is active
                idx_in_progress = torch.nonzero(~done & (current_player != initial_player), as_tuple=True)[0]
                # => (B') of indexes

            if idx_in_progress.numel() == 0:
                break

            state_in_progress = state[idx_in_progress]
            state_logits_in_progress = state_logits[idx_in_progress]

            if i == 0:
                action_in_progress = action[idx_in_progress]
                if p10n_strategy == p10n.Prediction.PROBS:
                    action_probs_in_progress = F.one_hot(action_in_progress, num_classes=N_ACTIONS).float()
                idx_in_progress_valid_action = idx_in_progress
            else:
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

            # hexactions = action[action > 1]
            # hexactions = hexactions[hexactions % 14] != 12
            # if hexactions.numel() > 0:
            #     import ipdb; ipdb.set_trace()  # noqa
            #     pass

            s = state[idx_in_progress]
            reward[idx_in_progress] += 1000 * (
                s[:, self.index_enemy_value_lost_now_rel] - s[:, self.index_my_value_lost_now_rel]
                + self.reward_dmg_mult * (s[:, self.index_enemy_dmg_received_now_rel] - s[:, self.index_my_dmg_received_now_rel])
            )

        if idx_in_progress.numel() > 0:
            print(f"WARNING: state still in progress after {MAX_TRANSITIONS} transitions")

        idx_for_fixed_reward = torch.nonzero(~initial_done, as_tuple=True)[0]
        reward[idx_for_fixed_reward] += self.reward_step_fixed

        # Give one final step reward + a term reward for states that are NOW done
        idx_for_step_and_term_reward = torch.nonzero(done & ~initial_done, as_tuple=True)[0]
        s = state[idx_for_step_and_term_reward]
        reward[idx_for_step_and_term_reward] += 1000 * (
            s[:, self.index_enemy_value_lost_now_rel] - s[:, self.index_my_value_lost_now_rel]
            + self.reward_dmg_mult * (s[:, self.index_enemy_dmg_received_now_rel] - s[:, self.index_my_dmg_received_now_rel])
            + self.reward_term_mult * (s[:, self.index_enemy_value_lost_acc_rel0] - s[:, self.index_my_value_lost_acc_rel0])
        )

        if t10n_strategy == t10n.Reconstruction.PROBS:
            state = self.transition_model.reconstruct(state_logits, strategy=t10n.Reconstruction.GREEDY)

        if callback:
            # Latest state should have no action
            callback(state[0].numpy(), -1)

        return state, reward, done


# XXX: The mppo-style HexConv network arch could be used here
class ModelFreeModel(nn.Module):
    """
    The model free path of the I2A consists of a CNN [...] without the FC layers.
    """
    def __init__(self):
        super().__init__()
        self.output_size = 32

        self.network = nn.Sequential(
            nn.LazyLinear(16),
            nn.LeakyReLU(),
            nn.LazyLinear(16),
            nn.LeakyReLU(),
            nn.LazyLinear(self.output_size)
        )

    def forward(self, obs):
        return self.network(obs)


class RolloutEncoder(nn.Module):
    """
    (I2A paper)
    The rollout encoder processes each frame [...] with another identically sized CNN.
    The output of this CNN is then concatenated with the reward prediction [...].
    This feature is the input to an LSTM
    """

    def __init__(self, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.imagination_core_model = ImaginationCoreModel()
        self.obs_processor = ModelFreeModel()
        self.lstm = nn.LSTM(
            input_size=1 + self.obs_processor.output_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, obs, action):
        assert len(obs.shape) == 2
        B = obs.size(0)
        T = self.horizon
        r_obs = obs.new_zeros(B, T, obs.size(1))
        r_rew = obs.new_zeros(B, T, 1)

        first = True
        for t in range(T):
            action = action if first else self.rollout_policy()
            first = False
            obs, rew = self.imagination_core_model(obs, action)
            r_obs[:, t, :] = obs
            r_rew[:, t, 0] = rew

        lstm_in = torch.cat((
            self.obs_processor(r_obs.flatten(end_dim=1)).unflatten(0, [B, T]),
            rew.unsqueeze(-1)
        ), dim=-1)

        """
        (I2A paper)
        The features are fed to the LSTM in reverse order [...].
        The choice of forward, backward or bi-directional processing seems to have
        relatively little impact on the performance of the I2A [...].
        """
        _out, (h_n, _c_n) = self.lstm(lstm_in.flip(1))

        """
        (Chat GPT; no info in paper)
        Last hidden state (h_n): one vector per rollout: use for I2A’s rollout embedding.
        Output sequence (out): per-step vectors: use only if you need fine‐grained, time-wise features.
        """

        return h_n


class ImaginationAggregatorModel(nn.Module):
    def __init__(self, num_trajectories, horizon):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.rollout_dim = 16
        self.rollout_encoder = RolloutEncoder(horizon, self.rollout_dim)

        # Attention-based aggregator
        self.query = nn.Parameter(torch.randn(1, 1, self.rollout_dim))
        self.mha = nn.MultiheadAttention(embed_dim=self.rollout_dim, num_heads=4, batch_first=True)

    def forward(self, obs, mask):
        B, A = mask.shape
        # 1) Count valid actions per row
        valid_counts = mask.sum(dim=1, keepdim=True)  # => (B, 1)
        # 2) Build a uniform probability distribution over valid actions
        probs = mask.float() / valid_counts  # => (B, A)
        # 3) Sample N actions; replacement=True if min(valid_counts) < N
        replacement = (valid_counts.min().item() < self.num_trajectories)
        actions = torch.multinomial(probs, self.num_trajectories, replacement=replacement)
        # => (B, N)

        rollouts = self.rollout_encoder(obs, actions)
        # => (B, N, X)

        q = self.query.expand(rollouts.size(0), 1, self.rollout_dim)
        attn_output, _ = self.mha(query=q, key=rollouts, value=rollouts)
        # => (B,1,X)

        return attn_output.squeeze(1)
        # => (B, X)


class I2AModel(nn.Module):
    def __init__(self, num_trajectories, horizon):
        self.model_free_model = ModelFreeModel()
        self.imagination_aggregator_model = ImaginationAggregatorModel(
            num_trajectories=num_trajectories,
            horizon=horizon,
        )

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
