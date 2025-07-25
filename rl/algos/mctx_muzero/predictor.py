import jax
import jax.numpy as jnp
import haiku as hk
import os

from jax import lax

import rl.world.t10n.jax.haiku.t10n as t10n_obs
import rl.world.t10n.jax.haiku.reward as t10n_rew
import rl.world.p10n.jax.haiku.p10n as p10n_act

from rl.world.util.constants_v12 import (
    STATE_SIZE,
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
)

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


class Predictor(hk.Module):
    def __init__(
        self,
        max_transitions: int,
        side: int,
        reward_dmg_mult: float,
        reward_term_mult: float,
        # TODO: add reward_step_fixed and use it

        jit: bool = False,
        name: str = None,
    ):
        super().__init__(name=name)
        self.max_transitions = max_transitions
        self.side = side
        self.reward_dmg_mult = reward_dmg_mult
        self.reward_term_mult = reward_term_mult
        self.jit = jit

        self.obs_model = t10n_obs.HaikuTransitionModel(deterministic=True, name="obs_model")
        self.rew_model = t10n_rew.HaikuTransitionModel(deterministic=True, name="rew_model")
        self.act_model = p10n_act.HaikuActionPredictionModel(deterministic=True, name="act_model")

        if self.side == 0:
            self.index_my_dmg_received_now_rel = INDEX_PLAYER0_DMG_RECEIVED_NOW_REL
            self.index_my_value_lost_now_rel = INDEX_PLAYER0_VALUE_LOST_NOW_REL
            self.index_my_value_lost_acc_rel0 = INDEX_PLAYER0_VALUE_LOST_ACC_REL0

            self.index_enemy_dmg_received_now_rel = INDEX_PLAYER1_DMG_RECEIVED_NOW_REL
            self.index_enemy_value_lost_now_rel = INDEX_PLAYER1_VALUE_LOST_NOW_REL
            self.index_enemy_value_lost_acc_rel0 = INDEX_PLAYER1_VALUE_LOST_ACC_REL0
        elif self.side == 1:
            self.index_my_dmg_received_now_rel = INDEX_PLAYER1_DMG_RECEIVED_NOW_REL
            self.index_my_value_lost_now_rel = INDEX_PLAYER1_VALUE_LOST_NOW_REL
            self.index_my_value_lost_acc_rel0 = INDEX_PLAYER1_VALUE_LOST_ACC_REL0

            self.index_enemy_dmg_received_now_rel = INDEX_PLAYER0_DMG_RECEIVED_NOW_REL
            self.index_enemy_value_lost_now_rel = INDEX_PLAYER0_VALUE_LOST_NOW_REL
            self.index_enemy_value_lost_acc_rel0 = INDEX_PLAYER0_VALUE_LOST_ACC_REL0
        else:
            raise Exception("Unknown side: %s" % self.side)

        if self.jit:
            self.predict_obs = jax.jit(self.obs_model.predict_batch)
            self.predict_rew = jax.jit(self.rew_model.predict_batch)
            self.predict_act = jax.jit(self.act_model.predict_batch)
        else:
            self.predict_obs = self.obs_model.predict_batch
            self.predict_rew = self.rew_model.predict_batch
            self.predict_act = self.act_model.predict_batch

    def setup_params(self, initial_state, initial_action):
        self.obs_model.predict_batch(initial_state, initial_action)
        self.rew_model.predict_batch(initial_state, initial_action)
        self.act_model.predict_batch(initial_state)

    def __call__(self, initial_state, initial_action):
        """Vectorised dream roll‑out, fully JIT‑compatible.

        Special cases handled:
          • *first_step* — player‑swap termination is ignored on t = 0.
          • *alternate win conditions* — army wiped out either before or after an
            action ("p10n –1") ends the episode and sets the winner flags.
        """

        B = initial_state.shape[0]

        # ------------------------------------------------------------------
        # helpers
        # ------------------------------------------------------------------

        def _terminated_now(state, finished):
            """Active‑player==NA or explicit winner flag set."""
            cur_player = state[:, INDEX_BSAP_START:INDEX_BSAP_END].argmax(1)
            cur_winner = state[:, INDEX_WINNER_START:INDEX_WINNER_END].argmax(1)
            return (~finished) & ((cur_player == 0) | (cur_winner > 0)), cur_player

        def _alt_dead_masks(state, finished):
            """Detect army‑wiped‑out deaths for both players."""
            state_hexes = state[:, HEXES_OFFSET:].reshape(-1, 165, STATE_SIZE_ONE_HEX)

            p0_alive = (
                (state[:, INDEX_PLAYER0_ARMY_VALUE_NOW_ABS] > 0)
                & (state[:, INDEX_PLAYER0_ARMY_VALUE_NOW_REL] > 0)
                & (state_hexes[:, :, INDEX_HEX_STACK_SIDE_PLAYER0].sum(1) > 0)
            )
            p1_alive = (
                (state[:, INDEX_PLAYER1_ARMY_VALUE_NOW_ABS] > 0)
                & (state[:, INDEX_PLAYER1_ARMY_VALUE_NOW_REL] > 0)
                & (state_hexes[:, :, INDEX_HEX_STACK_SIDE_PLAYER1].sum(1) > 0)
            )
            p0_dead = (~finished) & (~p0_alive)
            p1_dead = (~finished) & (~p1_alive)
            return p0_dead, p1_dead

        # ------------------------------------------------------------------
        # scan body
        # ------------------------------------------------------------------

        def step(carry, _):
            (
                state,
                action,
                reward,
                terminated,
                finished,
                num_t,
                first_step,  # scalar bool (0‑D array)
            ) = carry

            # --------------------------------------------------------------
            # 1. env‑level terminations (winner flag / no active player)
            # --------------------------------------------------------------
            term_now, cur_player = _terminated_now(state, finished)

            # --------------------------------------------------------------
            # 2. alternate win conditions BEFORE acting (army wiped out)
            # --------------------------------------------------------------
            p0_dead, p1_dead = _alt_dead_masks(state, finished)
            alt_term_pre = p0_dead | p1_dead

            # mark winners & NA active player where relevant
            state = state.at[:, INDEX_WINNER_PLAYER1].set(jnp.where(p0_dead, 1, state[:, INDEX_WINNER_PLAYER1]))
            state = state.at[:, INDEX_BSAP_PLAYER_NA].set(jnp.where(p1_dead, 1, state[:, INDEX_BSAP_PLAYER_NA]))
            state = state.at[:, INDEX_BSAP_PLAYER_NA].set(jnp.where(alt_term_pre, 1, state[:, INDEX_BSAP_PLAYER_NA]))

            # --------------------------------------------------------------
            # 3. player‑swap termination (except first step)
            # --------------------------------------------------------------
            swap_now = (cur_player == initial_player) & (~first_step)

            # aggregate new terminations / finishes
            fin_now = term_now | alt_term_pre | swap_now
            terminated |= term_now | alt_term_pre
            finished |= fin_now

            # --------------------------------------------------------------
            # 4. choose next action for unfinished trajectories
            # --------------------------------------------------------------
            next_action = lax.select(
                finished,
                action,
                self.predict_act(state).astype(jnp.int32),
            )

            # --------------------------------------------------------------
            # 5. p10n model signals end with action == −1 (post‑action alt win)
            # --------------------------------------------------------------
            terminated_by_action = (~finished) & (next_action == -1)
            p0_dead_a = terminated_by_action & (
                state[:, INDEX_PLAYER0_ARMY_VALUE_NOW_REL] < state[:, INDEX_PLAYER1_ARMY_VALUE_NOW_REL]
            )
            p1_dead_a = terminated_by_action & (~p0_dead_a)

            # update winners / active player flags
            state = state.at[:, INDEX_WINNER_PLAYER1].set(jnp.where(p0_dead_a, 1, state[:, INDEX_WINNER_PLAYER1]))
            state = state.at[:, INDEX_BSAP_PLAYER_NA].set(jnp.where(p1_dead_a, 1, state[:, INDEX_BSAP_PLAYER_NA]))
            state = state.at[:, INDEX_BSAP_PLAYER_NA].set(jnp.where(terminated_by_action, 1, state[:, INDEX_BSAP_PLAYER_NA]))

            terminated |= terminated_by_action
            finished |= terminated_by_action

            # --------------------------------------------------------------
            # 6. rewards and transition to next observation
            # --------------------------------------------------------------
            add_rew = self.predict_rew(state, next_action)
            reward += lax.select(finished, jnp.zeros_like(add_rew), add_rew)

            next_state = jnp.where(
                finished[:, None],
                state,
                self.predict_obs(state, next_action).astype(jnp.float32),
            ).astype(jnp.float32)

            num_t += (~finished).astype(jnp.int32)

            carry_out = (
                next_state,
                next_action,
                reward,
                terminated,
                finished,
                num_t,
                jnp.bool_(False),  # subsequent iterations are no longer first
            )
            return carry_out, None

        # ------------------------------------------------------------------
        # initial bookkeeping
        # ------------------------------------------------------------------

        initial_player = initial_state[:, INDEX_BSAP_START:INDEX_BSAP_END].argmax(1)
        initial_winner = initial_state[:, INDEX_WINNER_START:INDEX_WINNER_END].argmax(1)
        initial_terminated = (initial_player == 0) | (initial_winner > 0)

        carry0 = (
            initial_state,
            initial_action,
            jnp.zeros((B,), dtype=jnp.float32),  # reward so far
            initial_terminated,
            initial_terminated,
            jnp.zeros((B,), dtype=jnp.int32),    # num_t
            jnp.bool_(True),                     # first_step flag
        )

        # ------------------------------------------------------------------
        # roll‑out loop (static length, fully on device)
        # ------------------------------------------------------------------

        carry_final, _ = lax.scan(
            step,
            carry0,
            xs=None,
            length=self.max_transitions - 1,
        )

        (
            state_f,
            _action_f,
            reward_f,
            terminated_f,
            _finished_f,
            num_t_f,
            _,
        ) = carry_final

        # ------------------------------------------------------------------
        # terminal reward (vectorised)
        # ------------------------------------------------------------------

        term_this_phase = terminated_f & (~initial_terminated)
        extra = (
            state_f[:, self.index_enemy_value_lost_now_rel] - state_f[:, self.index_my_value_lost_now_rel]
            + self.reward_dmg_mult * (state_f[:, self.index_enemy_dmg_received_now_rel] - state_f[:, self.index_my_dmg_received_now_rel])
            + self.reward_term_mult * (state_f[:, self.index_enemy_value_lost_acc_rel0] - state_f[:, self.index_my_value_lost_acc_rel0])
        )
        reward_f = reward_f + jnp.where(term_this_phase, 1000 * extra, 0.0)

        return state_f, reward_f, terminated_f, num_t_f


class PredictorWrapper:
    def __init__(
        self,
        max_transitions: int,
        side: int,
        reward_dmg_mult: float,
        reward_term_mult: float,
        # TODO: add reward_step_fixed and use it

        jit: bool = False,
        name: str = None,
    ):
        self.max_transitions = max_transitions
        self.side = side
        self.reward_dmg_mult = reward_dmg_mult
        self.reward_term_mult = reward_term_mult
        self.jit = jit

        model_kwargs = {
            "max_transitions": max_transitions,
            "side": side,
            "reward_dmg_mult": reward_dmg_mult,
            "reward_term_mult": reward_term_mult,
            "jit": jit,
            "name": name,
        }

        def forward_fn(initial_state, initial_action):
            model = Predictor(**model_kwargs)
            return model(initial_state, initial_action)

        def setup_params_fn(initial_state, initial_action):
            model = Predictor(**model_kwargs)
            return model.setup_params(initial_state, initial_action)

        self.predictor_fwd = hk.transform(forward_fn)
        self.predictor_setup_params = hk.transform(setup_params_fn)

        # create a jitted apply that compiles once and reuses the XLA binary
        @jax.jit
        def jit_fwd(params, rng, obs, action):
            return self.predictor_fwd.apply(params, rng, obs, action)

        self.jit_fwd = jit_fwd

    # Get initial params (wrapper around hk.Model#init)
    def init(self, rng, initial_state, initial_action):
        return self.predictor_setup_params.init(rng, initial_state, initial_action)

    # Returns params loaded from torch files (custom method)
    def load(self, params, rng):
        import torch
        torch_state_obs = torch.load("hauzybxn-model.pt", weights_only=True, map_location="cpu")
        torch_state_rew = torch.load("aexhrgez-model.pt", weights_only=True, map_location="cpu")
        torch_state_act = torch.load("ogyesvkb-model.pt", weights_only=True, map_location="cpu")

        params = hk.data_structures.to_mutable_dict(params)

        from rl.world.t10n.jax.haiku.load_utils import load_params_from_torch_state
        prefix = "haiku_transition_model"

        prefix_obs = "predictor/~/obs_model"
        params_obs = {k.replace(prefix_obs, prefix, 1): v for k, v in params.items() if k.startswith(prefix_obs)}
        params_obs = load_params_from_torch_state(params_obs, torch_state_obs, head_names=["global", "player", "hex"])

        prefix_rew = "predictor/~/rew_model"
        params_rew = {k.replace(prefix_rew, prefix, 1): v for k, v in params.items() if k.startswith(prefix_rew)}
        params_rew = load_params_from_torch_state(params_rew, torch_state_rew, head_names=["reward"])

        prefix_act = "predictor/~/act_model"
        params_act = {k.replace(prefix_act, prefix, 1): v for k, v in params.items() if k.startswith(prefix_act)}
        params_act = load_params_from_torch_state(params_act, torch_state_act, head_names=["main", "hex"])

        return hk.data_structures.to_immutable_dict({
            **{k.replace(prefix, prefix_obs, 1): v for k, v in params_obs.items()},
            **{k.replace(prefix, prefix_rew, 1): v for k, v in params_rew.items()},
            **{k.replace(prefix, prefix_act, 1): v for k, v in params_act.items()}
        })

    # Forward pass (wrapper around hk.Model#apply)
    def apply(self, params, rng, initial_state, initial_action):
        func = self.jit_fwd if self.jit else self.predictor_fwd.apply
        return func(params, rng, initial_state, initial_action)


if __name__ == "__main__":
    env_kwargs = dict(
        mapname="gym/generated/evaluation/8x512.vmap",
        opponent="BattleAI",
        role="defender",
        swap_sides=0,
        random_heroes=1,
        random_obstacles=0,
        town_chance=0,
        warmachine_chance=0,
        random_terrain_chance=0,
        random_stack_chance=0,
        tight_formation_chance=0,
        reward_step_fixed=-0.01,
        reward_dmg_mult=0.01,
        reward_term_mult=0.01,
        reward_relval_mult=0.01,
    )

    # always enable "test_cuda" on my mac (for dev purposes)
    have_cuda = any(device.platform == 'gpu' for device in jax.devices())
    test_cuda = os.getenv("USER", "") == "simo" or have_cuda

    modelw = PredictorWrapper(
        jit=True,
        max_transitions=5,
        side=(env_kwargs["role"] == "defender"),
        reward_dmg_mult=env_kwargs["reward_dmg_mult"],
        reward_term_mult=env_kwargs["reward_term_mult"],
    )

    rng = jax.random.PRNGKey(0)
    params = modelw.init(rng, initial_state=jnp.zeros([1, STATE_SIZE]), initial_action=jnp.array([1]))
    params = modelw.load(params, rng)

    if test_cuda:
        import torch
        from rl.world.i2a import ImaginationCore
        torch_model = ImaginationCore(
            max_transitions=modelw.max_transitions,
            side=modelw.side,
            reward_step_fixed=0,
            reward_dmg_mult=modelw.reward_dmg_mult,
            reward_term_mult=modelw.reward_term_mult,
            transition_model_file="hauzybxn-model.pt",
            reward_prediction_model_file="aexhrgez-model.pt",
            action_prediction_model_file="ogyesvkb-model.pt",
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

    # Initialize env AFTER model (changes cwd)
    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    env = VcmiEnv(**env_kwargs)
    env.reset()

    # TEST

    buffer = {"act": [], "obs": [], "rew": [], "term": []}

    nsteps = 1000 if have_cuda else 10
    nsplits = 10
    assert nsteps % nsplits == 0

    for _ in range(nsteps):
        if env.terminated or env.truncated:
            env.reset()
        act = env.random_action()
        obs, rew, term, trunc, _info = env.step(act)

        buffer["act"].append(act)
        buffer["obs"].append(obs)
        buffer["rew"].append(rew)
        buffer["term"].append(term)

    import numpy as np
    split_obs = np.array(np.split(np.array([o["observation"] for o in buffer["obs"]]), nsplits))
    split_act = np.array(np.split(np.array(buffer["act"]), nsplits))

    import time

    # warmup
    modelw.apply(params, rng, jnp.array(split_obs[0]), jnp.array(split_act[0]))
    with torch.no_grad():
        torch_model(torch.as_tensor(split_obs[0], device=torch_model.device), torch.as_tensor(split_act[0], device=torch_model.device, dtype=torch.int64))

    if test_cuda:
        print("------- BATCH TEST JAX ON CUDA ----------")

        print("Benchmarking jax (%dx%d)..." % (nsplits, len(split_act[0])))
        batch_start = time.perf_counter()
        for b_obs, b_act in zip(split_obs, split_act):
            modelw.apply(params, rng, jnp.array(b_obs), jnp.array(b_act))
            print(".", end="", flush=True)
        batch_end = time.perf_counter()
        print("\ntime: %.2fs" % (batch_end - batch_start))

        # cuda_obs = torch.as_tensor(np.array([o["observation"] for o in buffer["obs"]]), device=torch_model.device)
        # cuda_act = torch.as_tensor(buffer["act"], device=torch_model.device, dtype=torch.int64)

        print("------- BATCH TEST TORCH ON CUDA ----------")
        print("Benchmarking cuda (%dx%d)..." % (nsplits, len(split_act[0])))
        with torch.no_grad():
            batch_start = time.perf_counter()
            for b_obs, b_act in zip(split_obs, split_act):
                torch_model(
                    initial_state=torch.as_tensor(b_obs, device=torch_model.device),
                    initial_action=torch.as_tensor(b_act, device=torch_model.device, dtype=torch.int64),
                )
                print(".", end="", flush=True)
        batch_end = time.perf_counter()
        print("\ntime: %.2fs" % (batch_end - batch_start))

    import ipdb; ipdb.set_trace()  # noqa

    # pdb:
    # from vcmi_gym.envs.v12.decoder.decoder import Decoder
    # i=1; obs, act, jres, tres = split_obs[i], split_act[i], jit_fwd(params, jnp.array(split_obs[i]), jnp.array(split_act[i])), torch_model(torch.as_tensor(split_obs[i], device=torch_model.device), torch.as_tensor(split_act[i], device=torch_model.device, dtype=torch.int64))
    # print("Action: ", act[0]); print(Decoder.decode(obs[0]).render(0)); print("------------------- Torch:"); print(Decoder.decode(np.array(tres[0][0].detach())).render(0)); print("---------------------- JAX:"); print(Decoder.decode(np.array(jres[0][0])).render(0))

    episodes = 0
    step = 0

    act = buffer["act"][0]
    obs0 = buffer["obs"][0]
    rew = buffer["rew"][0]
    term = buffer["term"][0]

    num_transitions = len(obs0["transitions"]["observations"])
    # if num_transitions < 3:
    #     continue

    print("Step: %d" % step)

    # if num_transitions != 4:
    #     continue

    start_obs = obs0["transitions"]["observations"][0]
    start_act = obs0["transitions"]["actions"][0]

    print("=" * 100)
    # env.render_transitions(add_regular_render=False)
    print("^ Transitions: %d" % num_transitions)
    # print("Dream act: %s" % start_act)

    state, reward, done, num_t = modelw.apply(
        params,
        rng,
        initial_state=jnp.expand_dims(start_obs, axis=0),
        initial_action=jnp.array([start_act])
    )

    # render_dream(dream)
    print("Predicted transitions: %s, real: %s" % (num_t, num_transitions-1))
    print("[GREEDY] Done: %s" % str([done, done[0]]))
    print("[GREEDY] Reward: %s" % str([rew, reward[0]]))

    from vcmi_gym.envs.v12.decoder.decoder import Decoder
    import numpy as np

    print("Start obs:")
    print(Decoder.decode(np.array(start_obs)).render(start_act))
    print("Next obs:")
    print(Decoder.decode(np.array(obs0["transitions"]["observations"][-1])).render(0))

    print("Predicted obs:")
    print(Decoder.decode(np.array(state)[0]).render(0))

    import ipdb; ipdb.set_trace()  # noqa
    pass


