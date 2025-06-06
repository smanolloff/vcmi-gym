import torch
from torch import nn
import torch.nn.functional as F

from .t10n import t10n, reward
from .p10n import p10n_nll as p10n

from .util.misc import layer_init
from .util.hexconv import HexConvResBlock
from .util.constants_v12 import (
    STATE_SIZE,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
    N_ACTIONS,
)

from .util.timer import Timer

TIMER_ALL = Timer()
TIMERS = {
    "ia": Timer(),
    "ic": Timer(),
    "ic.p10n": Timer(),
    "ic.t10n": Timer(),
    "ic.rew": Timer(),
    "ic.t10n.reconstruct": Timer(),
    "lstm": Timer(),
    "mf": Timer(),
    "re": Timer(),
    "re.get_action": Timer(),
}

INDEX_BSAP_START = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][1]
INDEX_BSAP_END = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][2] + INDEX_BSAP_START
assert GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][2] == 3  # N/A, P0, P1
INDEX_BSAP_PLAYER_NA = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][1]

INDEX_WINNER_START = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1]
INDEX_WINNER_END = GLOBAL_ATTR_MAP["BATTLE_WINNER"][2] + INDEX_WINNER_START
assert GLOBAL_ATTR_MAP["BATTLE_WINNER"][2] == 3  # N/A, P0, P1
INDEX_WINNER_PLAYER0 = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1] + 1
INDEX_WINNER_PLAYER1 = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1] + 2

HEXES_OFFSET = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER

P_OFFSET0 = STATE_SIZE_GLOBAL
P_OFFSET1 = STATE_SIZE_GLOBAL + STATE_SIZE_ONE_PLAYER

# For reward calculation:
INDEX_PLAYER0_DMG_RECEIVED_NOW_REL = P_OFFSET0 + PLAYER_ATTR_MAP["DMG_RECEIVED_NOW_REL"][1]
INDEX_PLAYER1_DMG_RECEIVED_NOW_REL = P_OFFSET1 + PLAYER_ATTR_MAP["DMG_RECEIVED_NOW_REL"][1]
INDEX_PLAYER0_VALUE_LOST_NOW_REL = P_OFFSET0 + PLAYER_ATTR_MAP["VALUE_LOST_NOW_REL"][1]
INDEX_PLAYER1_VALUE_LOST_NOW_REL = P_OFFSET1 + PLAYER_ATTR_MAP["VALUE_LOST_NOW_REL"][1]
INDEX_PLAYER0_VALUE_LOST_ACC_REL0 = P_OFFSET0 + PLAYER_ATTR_MAP["VALUE_LOST_ACC_REL0"][1]
INDEX_PLAYER1_VALUE_LOST_ACC_REL0 = P_OFFSET1 + PLAYER_ATTR_MAP["VALUE_LOST_ACC_REL0"][1]

# For constructing action mask:
INDEX_HEX_ACTION_MASK_START = HEX_ATTR_MAP["ACTION_MASK"][1]
INDEX_HEX_ACTION_MASK_END = HEX_ATTR_MAP["ACTION_MASK"][2] + INDEX_HEX_ACTION_MASK_START

INDEX_GLOBAL_ACTION_MASK_START = GLOBAL_ATTR_MAP["ACTION_MASK"][1]
INDEX_GLOBAL_ACTION_MASK_END = GLOBAL_ATTR_MAP["ACTION_MASK"][2] + INDEX_GLOBAL_ACTION_MASK_START

# For alternate loss conditions:
INDEX_PLAYER0_ARMY_VALUE_NOW_REL = P_OFFSET0 + PLAYER_ATTR_MAP["ARMY_VALUE_NOW_REL"][1]
INDEX_PLAYER1_ARMY_VALUE_NOW_REL = P_OFFSET1 + PLAYER_ATTR_MAP["ARMY_VALUE_NOW_REL"][1]
INDEX_PLAYER0_ARMY_VALUE_NOW_ABS = P_OFFSET0 + PLAYER_ATTR_MAP["ARMY_VALUE_NOW_ABS"][1]
INDEX_PLAYER1_ARMY_VALUE_NOW_ABS = P_OFFSET1 + PLAYER_ATTR_MAP["ARMY_VALUE_NOW_ABS"][1]

# relative to hex
assert HEX_ATTR_MAP["STACK_SIDE"][2] == 3  # N/A, P0, P1
INDEX_HEX_STACK_SIDE_PLAYER0 = HEX_ATTR_MAP["STACK_SIDE"][1] + 1
INDEX_HEX_STACK_SIDE_PLAYER1 = HEX_ATTR_MAP["STACK_SIDE"][1] + 2

assert HEX_ATTR_MAP["STACK_FLAGS1"][0].endswith("_ZERO_NULL")
INDEX_HEX_STACK_IS_ACTIVE = HEX_ATTR_MAP["STACK_FLAGS1"][1]  # 1st flag = IS_ACTIVE

assert HEX_ATTR_MAP["STACK_QUEUE"][0].endswith("_ZERO_NULL")
INDEX_HEX_STACK_STACK_QUEUE = HEX_ATTR_MAP["STACK_QUEUE"][1]  # 1st bit = currently active


class ImaginationCore(nn.Module):
    def __init__(
        self,
        side,
        reward_step_fixed,
        reward_dmg_mult,
        reward_term_mult,
        max_transitions,
        transition_model_file,
        action_prediction_model_file,
        reward_prediction_model_file,
        device=torch.device("cpu"),
    ):
        super().__init__()

        def load_weights(model, file):
            model.load_state_dict(torch.load(file, weights_only=True, map_location=device), strict=True)

        self.device = device
        self.side = side
        self.reward_step_fixed = torch.tensor(reward_step_fixed, dtype=torch.float32, device=device)
        self.reward_dmg_mult = torch.tensor(reward_dmg_mult, dtype=torch.float32, device=device)
        self.reward_term_mult = torch.tensor(reward_term_mult, dtype=torch.float32, device=device)
        self.max_transitions = max_transitions
        self.num_truncations = 0

        self.transition_model = t10n.TransitionModel(device)
        self.transition_model.eval()

        self.action_prediction_model = p10n.ActionPredictionModel(device)
        self.action_prediction_model.eval()

        self.reward_prediction_model = reward.TransitionModel(device)
        self.reward_prediction_model.eval()

        load_weights(self.transition_model, transition_model_file)
        load_weights(self.action_prediction_model, action_prediction_model_file)
        load_weights(self.reward_prediction_model, reward_prediction_model_file)
        self._layer_initialized = True  # prevents layer_init() from overwriting loaded weights

        assert HEX_ATTR_MAP["STACK_SIDE"][2] == 3  # NULL, RED, BLUE

        # These are single-value attributes (i.e. no explicit nulls => n=1)
        assert PLAYER_ATTR_MAP["ARMY_VALUE_NOW_ABS"][2] == 1
        assert PLAYER_ATTR_MAP["ARMY_VALUE_NOW_REL"][2] == 1
        assert PLAYER_ATTR_MAP["DMG_RECEIVED_NOW_REL"][2] == 1
        assert PLAYER_ATTR_MAP["VALUE_LOST_NOW_REL"][2] == 1
        assert PLAYER_ATTR_MAP["VALUE_LOST_ACC_REL0"][2] == 1
        # These are permille values (i.e. vmax=1000)
        assert PLAYER_ATTR_MAP["ARMY_VALUE_NOW_REL"][3] == 1000
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

    def forward(
        self,
        initial_state,
        initial_action,
        t10n_strategy=t10n.Reconstruction.GREEDY,
        p10n_strategy=p10n.Prediction.GREEDY,
        callback=None,
        obs0=None,
        debug=False
    ):
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

        # Use to to simulate 100% accurate prediction
        # if debug and obs0:
        #     debugstate = {"i": 0}
        #     def debugtransition(*args, **kwargs):
        #         debugstate["i"] += 1
        #         return torch.as_tensor(obs0["transitions"]["observations"][debugstate["i"]]).unsqueeze(0).clone()
        #     # tm = self.transition_model
        #     tm = debugtransition
        B = initial_state.size(0)
        num_t = torch.zeros(B, dtype=torch.long, device=self.device)

        if debug:
            # Every batch will have different num_transitions
            action_hist = torch.zeros(B, self.max_transitions, dtype=torch.long, device=self.device).fill_(-1)
            done_hist = torch.zeros(B, self.max_transitions, dtype=torch.long, device=self.device).fill_(-1)
            state_hist = torch.zeros(B, self.max_transitions, initial_state.size(1), device=self.device).fill_(-1)
            state_logits_hist = state_hist.clone()
            action_hist[:, 0] = initial_action
            done_hist[:, 0] = initial_done
            state_hist[:, 0, :] = initial_state
            state_logits_hist[:, 0, :] = initial_state

        # Initial transition is always the input observation
        # => max-1 transitions to predict
        for t in range(0, self.max_transitions-1):
            if t == 0:
                current_player = initial_player
                current_winner = initial_winner
                action = initial_action.clone()
                done = initial_done.clone()
                state = initial_state.clone()
                state_logits = initial_state.clone()

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

                # check in_progress states for alternate win conditions
                # NOTE: this will probably not work if state is just probs (and not reconstructed)
                state_hexes_in_progress = state_in_progress[:, HEXES_OFFSET:].unflatten(1, [165, -1])
                # a.k.a. not(VALUE_ABS_P0 > 0 and VALUE_REL_P0 > 0 AND ANY(HEX_STACK_SIDE_P0 > 0))
                p0_alive = (
                    (state_in_progress[:, INDEX_PLAYER0_ARMY_VALUE_NOW_ABS] > 0)
                    & (state_in_progress[:, INDEX_PLAYER0_ARMY_VALUE_NOW_REL] > 0)
                    & (state_hexes_in_progress[:, :, INDEX_HEX_STACK_SIDE_PLAYER0].sum(dim=1) > 0)
                )
                idx_p0_dead = (~p0_alive).nonzero().unique()
                # => (B'') of indexes where P0 looks pretty dead
                p1_alive = (
                    (state_in_progress[:, INDEX_PLAYER1_ARMY_VALUE_NOW_ABS] > 0)
                    & (state_in_progress[:, INDEX_PLAYER1_ARMY_VALUE_NOW_REL] > 0)
                    & (state_hexes_in_progress[:, :, INDEX_HEX_STACK_SIDE_PLAYER1].sum(dim=1) > 0)
                )
                idx_p1_dead = (~p1_alive).nonzero().unique()
                # => (B'') of indexes where P1 looks pretty dead

                # Given idx_in_progress=[0,2,5,7] and idx_p0_dead=[1,3]
                # => real_idx_p0_dead=[2,7]
                real_idx_p0_dead = idx_in_progress[idx_p0_dead]
                real_idx_p1_dead = idx_in_progress[idx_p1_dead]

                if debug and (real_idx_p0_dead.numel() > 0 or real_idx_p1_dead.numel() > 0):
                    print("ALTERNATE WINCON")
                    import ipdb; ipdb.set_trace()  # noqa
                    pass

                # Mark winners and set active player to NA
                state[real_idx_p0_dead, INDEX_WINNER_PLAYER1] = 1
                state[real_idx_p0_dead, INDEX_BSAP_PLAYER_NA] = 1

                state[real_idx_p1_dead, INDEX_WINNER_PLAYER0] = 1
                state[real_idx_p1_dead, INDEX_BSAP_PLAYER_NA] = 1

                # Re-evaluate which batches are done
                done[real_idx_p0_dead] = True
                done[real_idx_p1_dead] = True
                idx_in_progress = torch.nonzero(~done & (current_player != initial_player), as_tuple=True)[0]

            state_in_progress = state[idx_in_progress]
            state_logits_in_progress = state_logits[idx_in_progress]

            if idx_in_progress.numel() == 0:
                break

            if t == 0:
                action_in_progress = action[idx_in_progress]
                if p10n_strategy == p10n.Prediction.PROBS:
                    action_probs_in_progress = F.one_hot(action_in_progress, num_classes=N_ACTIONS).float()
                idx_in_progress_valid_action = idx_in_progress
            else:
                with TIMERS["ic.p10n"]:
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
                with TIMERS["ic.t10n"]:
                    state_logits[idx_in_progress] = self.transition_model.forward_probs(state_in_progress, action_probs_in_progress).float()
                with TIMERS["ic.rew"]:
                    reward[idx_in_progress] += self.reward_prediction_model.forward_probs(state_in_progress, action_probs_in_progress).float()
            else:
                with TIMERS["ic.t10n"]:
                    state_logits[idx_in_progress] = self.transition_model(state_in_progress, action_in_progress).float()
                with TIMERS["ic.rew"]:
                    reward[idx_in_progress] += self.reward_prediction_model(state_in_progress, action_in_progress).float()

            with TIMERS["ic.t10n.reconstruct"]:
                state[idx_in_progress] = self.transition_model.reconstruct(state_logits[idx_in_progress], strategy=t10n_strategy)
            num_t[idx_in_progress] += 1

            if debug:
                done_hist[idx_in_progress, t] = done[idx_in_progress].long()
                action_hist[idx_in_progress, t] = action_in_progress.long()
                state_hist[idx_in_progress, t+1, :] = state[idx_in_progress]
                state_logits_hist[idx_in_progress, t+1, :] = state_logits[idx_in_progress]

            if callback:
                if t10n_strategy == t10n.Reconstruction.PROBS:
                    # Rendering probs will likely fail => collapse first
                    greedy = self.transition_model.reconstruct(state_logits_in_progress, strategy=t10n.Reconstruction.GREEDY)
                    callback(greedy[0].numpy(), action_in_progress.item())
                else:
                    # Rendering greedy is ok, samples is kind-of-ok => leave as-is
                    callback(state_in_progress[0].numpy(), action_in_progress.item())

        if idx_in_progress.numel() > 0:
            self.num_truncations += idx_in_progress.numel()
            # print(f"WARNING: state still in progress after {self.max_transitions} transitions")

            if debug:
                from vcmi_gym.envs.v12.decoder.decoder import Decoder

                def decode(b, t):
                    return Decoder.decode(state_hist[b, t].cpu().numpy())

                def render(b, t):
                    print(decode(b, t).render(action_hist[b, t].item()))

                import ipdb; ipdb.set_trace()  # noqa
                print("")

        # Give one final step reward + a term reward for states that are NOW done
        # The reward_prediction_model cannot predict terminal rewards
        # => calculate them instead
        idx_for_step_and_term_reward = torch.nonzero(done & ~initial_done, as_tuple=True)[0]
        s = state[idx_for_step_and_term_reward]  # don't use -1 (= MAX_TRANSITIONS)
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

        return state, reward, done, num_t


class ObsProcessor(nn.Module):
    """
    A shared network for model-free path and rollout encoder.
    For VCMI observations, a "HexConv" network might fit well here.
    (I2A paper):
    The model free path of the I2A consists of a CNN [...] without the FC layers.
    [...]
    The rollout encoder processes each frame [...] with another identically sized CNN.
    """
    def __init__(self, output_size):
        super().__init__()
        self.z_size_other = 64
        self.z_size_hex = 32
        self.output_size = output_size

        self.encoder_other = nn.Sequential(
            nn.LazyLinear(self.z_size_other),
            nn.LeakyReLU()
            # => (B, Z_OTHER)
        )

        self.encoder_hexes = nn.Sequential(
            # => (B, 165*H)
            nn.Unflatten(dim=1, unflattened_size=[165, STATE_SIZE_ONE_HEX]),
            # => (B, 165, H)
            HexConvResBlock(channels=STATE_SIZE_ONE_HEX, depth=3),
            # => (B, 165, H)
            nn.LazyLinear(out_features=self.z_size_hex),
            nn.LeakyReLU(),
            # => (B, 165, Z_HEX)
            nn.Flatten(),
            # => (B, 165*Z_HEX)
        )

        self.encoder_merged = nn.Sequential(
            # => (B, Z_OTHER + 165*Z_HEX)
            nn.LazyLinear(out_features=self.output_size),
            nn.LeakyReLU()
            # => (B, OUTPUT_SIZE)
        )

    def forward(self, obs, debug=False):
        other, hexes = torch.split(obs, [HEXES_OFFSET, 165*STATE_SIZE_ONE_HEX], dim=1)
        z_other = self.encoder_other(other)
        z_hexes = self.encoder_hexes(hexes)
        merged = torch.cat((z_other, z_hexes), dim=1)
        return self.encoder_merged(merged)


class RolloutEncoder(nn.Module):
    def __init__(
        self,
        rollout_dim,
        rollout_policy_fc_units,
        horizon,
        # Pass-through params (for ObsProcessor):
        obs_processor_output_size,
        # Pass-through params (for ImaginationCore):
        side,
        reward_step_fixed,
        reward_dmg_mult,
        reward_term_mult,
        max_transitions,
        transition_model_file,
        action_prediction_model_file,
        reward_prediction_model_file,
        device=torch.device("cpu"),
        debug_render=False,
    ):
        super().__init__()
        self.debug_render = debug_render

        self.rollout_dim = rollout_dim
        self.rollout_policy_fc_units = rollout_policy_fc_units
        self.horizon = horizon
        self.obs_processor = ObsProcessor(obs_processor_output_size)
        self.imagination_core = ImaginationCore(
            side=side,
            reward_step_fixed=reward_step_fixed,
            reward_dmg_mult=reward_dmg_mult,
            reward_term_mult=reward_term_mult,
            max_transitions=max_transitions,
            transition_model_file=transition_model_file,
            action_prediction_model_file=action_prediction_model_file,
            reward_prediction_model_file=reward_prediction_model_file,
            device=device,
        )

        """
        (I2A paper, section 3.1):
        We investigated several types of rollout policies [...] and found that
        a particularly efficient strategy was to distill the imagination-augmented policy
        into a [..] small model-free network πˆ(ot), and adding to the total loss a
        cross entropy auxiliary loss between the imagination-augmented policy π(ot)
        as computed on the current observation, and the policy πˆ(ot) as computed
        on the same observation.
        """
        self.rollout_policy = nn.Sequential(
            nn.LazyLinear(self.rollout_policy_fc_units),
            nn.LeakyReLU(),
            nn.LazyLinear(N_ACTIONS),
        )

        """
        (I2A paper, section B.1, under "I2A"):
        [...] an LSTM [...] used to process all [...] rollouts (one per action)
        """
        self.lstm = nn.LSTM(
            input_size=1 + self.obs_processor.output_size,
            hidden_size=self.rollout_dim,
            batch_first=True,
            # (I2A paper does not mention number of layers => assume 1)
            # num_layers=2,
            # dropout=0.2
        )

    def get_action(self, obs):
        hexes = obs[:, STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER:].unflatten(-1, [165, -1])
        hex_masks_logits = hexes[:, :, INDEX_HEX_ACTION_MASK_START:INDEX_HEX_ACTION_MASK_END]
        global_mask_logits = obs[:, INDEX_GLOBAL_ACTION_MASK_START:INDEX_GLOBAL_ACTION_MASK_END]

        mask_logits = torch.cat((global_mask_logits, hex_masks_logits.flatten(start_dim=1)), dim=-1)
        mask = mask_logits > 0
        mask[:, 0] = False
        # => (B, N_ACTIONS)

        # The rollout_policy is trained separately
        # (distilled from the I2A policy) => no grad needed here
        with torch.no_grad():
            action_logits = self.rollout_policy(obs)
        masked_logits = action_logits.masked_fill(~mask, torch.finfo(action_logits.dtype).min)
        probs = masked_logits.softmax(dim=-1)
        # => (B, N_ACTIONS)

        action = probs.multinomial(num_samples=1)
        # => (B, 1)

        return action.squeeze(1), mask
        # => (B), (B, N_ACTIONS)

    def render_step(self, state, action, mask, reward, done, headline):
        from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
        from vcmi_gym.envs.v12.decoder.decoder import Decoder
        import re
        bf = Decoder.decode(state)
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        rewtxt = "" if reward is None else "Reward: %s" % round(reward, 2)
        rewtxt += " (DONE)" if done else ""
        render = {}
        render["bf_lines"] = bf.render_battlefield()[0][:-1]
        render["bf_len"] = [len(l) for l in render["bf_lines"]]
        render["bf_printlen"] = [len(ansi_escape.sub('', l)) for l in render["bf_lines"]]
        render["bf_maxlen"] = max(render["bf_len"])
        render["bf_maxprintlen"] = max(render["bf_printlen"])
        render["bf_lines"].insert(0, rewtxt.ljust(render["bf_maxprintlen"]))
        render["bf_printlen"].insert(0, len(render["bf_lines"][0]))
        render["bf_lines"].insert(0, headline)
        render["bf_printlen"].insert(0, len(render["bf_lines"][0]))
        render["bf_lines"] = [l + " "*(render["bf_maxprintlen"] - pl) for l, pl in zip(render["bf_lines"], render["bf_printlen"])]
        render["bf_lines"].append(VcmiEnv.action_text(action, bf=bf).rjust(render["bf_maxprintlen"]))

        action_texts = [VcmiEnv.action_text(a, bf=bf) for a in mask.nonzero()[0]]
        short_actions = []
        for at in action_texts:
            m = re.match(r"Action (\d+): (.+) \(y=(\d+) x=(\d+)\)", at)
            if m:
                short = "%s:%s(%s,%s)" % m.groups()
            else:
                assert at == "Action 1: Wait", at
                short = "1:WAIT"

            short_actions.append(short)

        return render["bf_lines"], " ".join(short_actions)

    def render_dream(self, start_obs, start_act, start_mask, dream):
        al, ml = self.render_step(start_obs, start_act, start_mask, None, False, "Dream start")
        ary_lines = [al]
        mask_lines = [ml]

        for i, (s, a, m, r, d) in enumerate(dream):
            al, ml = self.render_step(s, a, m, r, d, "Dream step %d:" % i)
            ary_lines.append(al)
            mask_lines.append(ml)

        print("")
        print("\n".join([(" → ".join(rowlines)) for rowlines in zip(*ary_lines)]))
        for i, ml in enumerate(mask_lines):
            print("Step %d mask: %s" % (i, ml))
        print("")

    def forward(self, obs, action, mask, t10n_strategy, p10n_strategy, debug=False, render=False):
        assert len(obs.shape) == 2
        B = obs.size(0)
        T = self.horizon
        bt_obs = obs.new_zeros(B, T, obs.size(1))
        bt_rew = obs.new_zeros(B, T)
        bt_done = obs.new_zeros(B, T, dtype=torch.bool)

        if self.debug_render:
            start_obs = obs.detach()
            start_act = action.detach()
            start_mask = mask.detach()
            bt_act = action.new_zeros(B, T)
            bt_mask = torch.zeros(B, T, N_ACTIONS)

        for t in range(T):
            if t > 0:
                with TIMERS["re.get_action"]:
                    action, mask = self.get_action(obs)
                if self.debug_render:
                    bt_act[:, t-1] = action
                    bt_mask[:, t-1] = mask

            with torch.no_grad():
                with TIMERS["ic"]:
                    obs, rew, done, _length = self.imagination_core(
                        initial_state=obs,
                        initial_action=action,
                        t10n_strategy=t10n_strategy,
                        p10n_strategy=p10n_strategy,
                        debug=debug,
                    )

            bt_obs[:, t, :] = obs
            bt_rew[:, t] = rew
            bt_done[:, t] = done

        if self.debug_render:
            bt_act[:, T-1] = -1
            print("==========================================================")
            print("==========================================================")
            print("==========================================================")

            for b in range(B):
                print("------------------------------------------------------")
                self.render_dream(
                    start_obs[b].numpy(),
                    start_act[b].item(),
                    start_mask[b].numpy(),
                    [[bt_obs[b, t].numpy(), bt_act[b, t].item(), bt_mask[b, t].numpy(), bt_rew[b, t].item(), bt_done[b, t].item()] for t in range(T)]
                )
                import ipdb; ipdb.set_trace()  # noqa
                pass

        """
        (I2A paper, section B.1, under "I2A"):
        The output [...] is then concatenated with the reward prediction
        (single scalar broadcast into frame shape). This feature is the input
        to an LSTM [...] used to process all [...] rollouts
        """
        lstm_in = torch.cat((
            self.obs_processor(bt_obs.flatten(end_dim=1)).unflatten(0, [B, T]),
            bt_rew.unsqueeze(-1)
        ), dim=-1)
        # => (B, T, self.lstm.input_size)

        # Build a mask that stays True for every t ≥ the first done in that row
        # (so that when done=1 at time i, we mask i+1, i+2, …)
        post_term = bt_done.cumsum(dim=1) > 0  # (B, T) of flags where t >= (first done)
        lstm_mask = torch.zeros_like(bt_done)
        lstm_mask[:, 1:] = bt_done[:, :-1]
        lstm_in.masked_fill_(post_term.unsqueeze(-1), 0)

        """
        (I2A paper, section 3.2):
        The features are fed to the LSTM in reverse order [...].
        The choice of forward, backward or bi-directional processing seems to have
        relatively little impact on the performance of the I2A [...].
        """
        with TIMERS["lstm"]:
            _out, (h_n, _c_n) = self.lstm(lstm_in.flip(1))

        """
        (Chat GPT; no info in paper)
        Last hidden state (h_n): one vector per rollout: use for I2A’s rollout embedding.
        Output sequence (out): per-step vectors: use only if you need fine‐grained, time-wise features.
        """
        # XXX: h_n is always (1, B, X)
        return h_n.squeeze(0)


class ImaginationAggregator(nn.Module):
    def __init__(
        self,
        num_trajectories,
        # Pass-through params (for RolloutEncoder):
        rollout_dim,
        rollout_policy_fc_units,
        horizon,
        obs_processor_output_size,
        side,
        reward_step_fixed,
        reward_dmg_mult,
        reward_term_mult,
        max_transitions,
        transition_model_file,
        action_prediction_model_file,
        reward_prediction_model_file,
        device=torch.device("cpu"),
        debug_render=False,
    ):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.rollout_encoder = RolloutEncoder(
            rollout_dim=rollout_dim,
            rollout_policy_fc_units=rollout_policy_fc_units,
            horizon=horizon,
            obs_processor_output_size=obs_processor_output_size,
            side=side,
            reward_step_fixed=reward_step_fixed,
            reward_dmg_mult=reward_dmg_mult,
            reward_term_mult=reward_term_mult,
            max_transitions=max_transitions,
            transition_model_file=transition_model_file,
            action_prediction_model_file=action_prediction_model_file,
            reward_prediction_model_file=reward_prediction_model_file,
            device=device,
            debug_render=debug_render,
        )

        # Attention-based aggregator
        self.query = nn.Parameter(torch.randn(1, 1, self.rollout_encoder.rollout_dim))
        self.mha = nn.MultiheadAttention(embed_dim=self.rollout_encoder.rollout_dim, num_heads=4, batch_first=True)

    def forward(self, obs, mask, t10n_strategy, p10n_strategy, debug=False):
        B, A = mask.shape
        # 1) Count valid actions per row
        valid_counts = mask.sum(dim=1, keepdim=True)  # => (B, 1)
        # 2) Build a uniform probability distribution over valid actions
        probs = mask.float() / valid_counts  # => (B, A)
        # 3) Sample N actions; replacement=True if min(valid_counts) < N
        replacement = (valid_counts.min().item() < self.num_trajectories)

        # XXX: a runtime error "invalid multinomial distribution" will occur
        #      when mask allows no action (e.g. when input obs is terminal)
        actions = torch.multinomial(probs, self.num_trajectories, replacement=replacement)
        # => (B, N)

        actions = actions.flatten()
        # => (B*N)

        # XXX: clone is required to allow proper gradient flow
        obs = obs.unsqueeze(1).expand([B, self.num_trajectories, -1]).clone().flatten(end_dim=1)
        # => (B*N, STATE_SIZE)

        with TIMERS["re"]:
            rollouts = self.rollout_encoder(obs, actions, mask, t10n_strategy, p10n_strategy, debug).unflatten(0, [B, -1])
            # => (B, N, X)

        """
        (I2A paper, Appendix B.1, under "I2A"):
        The last output of the LSTM for all rollouts are concatenated into a
        single vector `cia` of length 2560 for Sokoban, and 1280 on MiniPacman
        """
        # XXX: multi-head attention will be used instead of concatenation
        # for a fixed-size output vector.
        # q = self.query.expand(rollouts.size(0), 1, self.rollout_encoder.rollout_dim)
        # attn_output, _ = self.mha(query=q, key=rollouts, value=rollouts)
        # => (B, 1, X)

        # return attn_output.squeeze(1)
        # => (B, X)

        return rollouts.flatten(start_dim=1)
        # => (B, N*X)
        # where N=num_trajectories, X=rollout_dim


class I2A(nn.Module):
    def __init__(
        self,
        i2a_fc_units,
        # Pass-through params (for ImaginationAggregator):
        num_trajectories,
        rollout_dim,
        rollout_policy_fc_units,
        horizon,
        obs_processor_output_size,
        side,
        reward_step_fixed,
        reward_dmg_mult,
        reward_term_mult,
        max_transitions,
        transition_model_file,
        action_prediction_model_file,
        reward_prediction_model_file,
        device=torch.device("cpu"),
        debug_render=False,
    ):
        super().__init__()
        self.imagination_aggregator = ImaginationAggregator(
            num_trajectories=num_trajectories,
            rollout_dim=rollout_dim,
            rollout_policy_fc_units=rollout_policy_fc_units,
            horizon=horizon,
            obs_processor_output_size=obs_processor_output_size,
            side=side,
            reward_step_fixed=reward_step_fixed,
            reward_dmg_mult=reward_dmg_mult,
            reward_term_mult=reward_term_mult,
            max_transitions=max_transitions,
            transition_model_file=transition_model_file,
            action_prediction_model_file=action_prediction_model_file,
            reward_prediction_model_file=reward_prediction_model_file,
            device=device,
            debug_render=debug_render,
        )

        self.device = device
        self.model_free_path = ObsProcessor(self.imagination_aggregator.rollout_encoder.obs_processor.output_size)

        """
        // XXX: The I2A paper is unclear regarding the FC layers.

        (I2A paper, Section 3.2):
        For the model-free path of the I2A, we chose a standard network of
        convolutional layers plus one fully connected one. . We also use this
        architecture on its own as a baseline agent.

        (I2A paper, Appendix B.1, under "I2A"):
        The model free path of the I2A consists of a CNN identical to one of
        the standard model-free baseline (without the FC layers)

        (I2A paper, Appendix B.1, under "Standard model-free baseline agent"):
            * for MiniPacman: [...] the following FC layer has 256 units
            * for Sokoban: [...] the following FC has 512 units
        """

        self.body = nn.Sequential(
            nn.LazyLinear(i2a_fc_units),
            nn.LeakyReLU()
        )

        self.action_head = nn.LazyLinear(N_ACTIONS)
        self.value_head = nn.LazyLinear(1)

        self.to(device)

        # Init lazy layers
        with torch.no_grad():
            obs = torch.randn([2, STATE_SIZE], device=device)
            mask = torch.randn([2, N_ACTIONS], device=device) > 0
            mask[:, 0] = False
            self.forward(obs, mask)

        layer_init(self)

    def forward(
        self,
        obs,
        mask,
        t10n_strategy=t10n.Reconstruction.GREEDY,
        p10n_strategy=p10n.Prediction.GREEDY,
        debug=False,
    ):
        with TIMER_ALL:
            with TIMERS["mf"]:
                c_mf = self.model_free_path(obs, debug)
            with TIMERS["ia"]:
                c_ia = self.imagination_aggregator(obs, mask, t10n_strategy, p10n_strategy, debug)
            z = self.body(torch.cat((c_ia, c_mf), dim=1))
            action_logits = self.action_head(z)
            value = self.value_head(z)

        print("=========================")
        print("Timers:")
        for k, v in TIMERS.items():
            print("%s: %.2f" % (k, v.peek() / TIMER_ALL.peek()))
            v.reset()
        TIMER_ALL.reset()

        return action_logits, value
