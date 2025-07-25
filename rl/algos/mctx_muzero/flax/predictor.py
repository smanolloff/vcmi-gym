import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as fnn
from flax.core import freeze, unfreeze

import rl.world.t10n.jax.flax.t10n as t10n_obs
import rl.world.t10n.jax.flax.reward as t10n_rew
import rl.world.p10n.jax.p10n as p10n_act

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


class Predictor(fnn.Module):
    max_transitions: int
    side: int
    reward_dmg_mult: float
    reward_term_mult: float
    # FIXME: add reward_step_fixed and use it

    # Very slow on CPU
    jit: bool = False

    def setup(self):
        self.obs_model = t10n_obs.FlaxTransitionModel(deterministic=True)
        self.rew_model = t10n_rew.FlaxTransitionModel(deterministic=True)
        self.act_model = p10n_act.FlaxActionPredictionModel(deterministic=True)

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

        # These make it a lot slower on CPU
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
        initial_player = initial_state[:, INDEX_BSAP_START:INDEX_BSAP_END].argmax(axis=1)
        # => (B)  # values 0=none, 1=red or 2=blue
        initial_winner = initial_state[:, INDEX_WINNER_START:INDEX_WINNER_END].argmax(axis=1)
        # => (B)  # values 0=none, 1=red or 2=blue
        initial_terminated = (initial_player == 0) | (initial_winner > 0)
        # => (B) of done flags

        action = initial_action
        reward = jnp.zeros((initial_state.shape[0],), dtype=initial_state.dtype)

        # => (B) of rewards

        final_state = initial_state
        B = initial_state.shape[0]
        num_t = jnp.zeros(B, dtype=jnp.int32)

        # Initial transition is always the input observation
        # => max-1 transitions to predict
        for t in range(0, self.max_transitions-1):
            if t == 0:
                # XXX:
                #   terminated -- episode ended
                #   finished -- episode OR transition ended
                #   *_now -- same as above, but it happened this `t`
                current_player = initial_player
                current_winner = initial_winner
                action = initial_action
                terminated_now = initial_terminated
                finished_now = initial_terminated
                terminated = initial_terminated
                finished = initial_terminated
                state = initial_state
            else:
                # state is (B, 28114)
                current_player = state[:, INDEX_BSAP_START:INDEX_BSAP_END].argmax(axis=1)
                current_winner = state[:, INDEX_WINNER_START:INDEX_WINNER_END].argmax(axis=1)
                terminated_now = ~finished & ((current_player == 0) | (current_winner > 0))
                finished_now = terminated_now | (current_player == initial_player)

                terminated |= terminated_now
                finished |= finished_now

                if finished_now.all():
                    # Since we always update all entries in `state`
                    # (even finished ones) through t10n and p10n,
                    # make sure to assign only those finished *now*
                    # XXX: must do this only if breaking from the loop,
                    #      otherwise they will re-assigned again later
                    if finished_now.any():
                        final_state = final_state.at[finished_now].set(state[finished_now])
                    break

                # check non-finished states for alternate win conditions
                # NOTE: this will probably not work if state is just probs (and not reconstructed)
                state_hexes = state[:, HEXES_OFFSET:].reshape(-1, 165, STATE_SIZE_ONE_HEX)

                # a.k.a. not(finished) AND not(alive)
                #   (alive means having value > 0 + at least 1 stack)
                p0_dead_mask = ~finished & ~(
                    (state[:, INDEX_PLAYER0_ARMY_VALUE_NOW_ABS] > 0)
                    & (state[:, INDEX_PLAYER0_ARMY_VALUE_NOW_REL] > 0)
                    & (state_hexes[:, :, INDEX_HEX_STACK_SIDE_PLAYER0].sum(axis=1) > 0)
                )
                # => (B) bool mask (True means P0 looks dead)

                p1_dead_mask = ~finished & ~(
                    (state[:, INDEX_PLAYER1_ARMY_VALUE_NOW_ABS] > 0)
                    & (state[:, INDEX_PLAYER1_ARMY_VALUE_NOW_REL] > 0)
                    & (state_hexes[:, :, INDEX_HEX_STACK_SIDE_PLAYER1].sum(axis=1) > 0)
                )
                # => (B) bool mask (True means P1 looks dead)

                terminated_now |= (p0_dead_mask | p1_dead_mask)
                finished_now |= terminated_now

                terminated |= terminated_now
                finished |= finished_now

                # Set active player to NA and mark winners
                state = state.at[terminated_now, INDEX_BSAP_PLAYER_NA].set(1)
                state = state.at[p0_dead_mask, INDEX_WINNER_PLAYER1].set(1)
                state = state.at[p1_dead_mask, INDEX_BSAP_PLAYER_NA].set(1)

            if finished.all():
                # XXX: must do this only if breaking from the loop (see comment above)
                if finished_now.any():
                    final_state = final_state.at[finished_now].set(state[finished_now])
                # import ipdb; ipdb.set_trace()  # noqa
                break

            if t != 0:
                action = action.at[~finished].set(self.predict_act(state[~finished]))

                # p10n predicts -1 when it believes battle has ended
                # => treat this as an additional alt termination condition
                # To determine who died, we must compare the army values...
                terminated_by_action = ~finished & (action == -1)

                p0_dead_mask = (
                    terminated_by_action
                    & (state[:, INDEX_PLAYER0_ARMY_VALUE_NOW_REL] < state[:, INDEX_PLAYER1_ARMY_VALUE_NOW_REL])
                )

                p1_dead_mask = terminated_by_action & (~p0_dead_mask)

                # Mark winners and set active player to NA
                state = state.at[p0_dead_mask, INDEX_WINNER_PLAYER1].set(1)
                state = state.at[p1_dead_mask, INDEX_BSAP_PLAYER_NA].set(1)
                state = state.at[terminated_by_action, INDEX_BSAP_PLAYER_NA].set(1)

                terminated_now |= terminated_by_action
                finished_now |= terminated_now

                terminated |= terminated_now
                finished |= finished_now

                if finished.all():
                    # XXX: must do this only if breaking from the loop (see comment above)
                    if finished_now.any():
                        final_state = final_state.at[finished_now].set(state[finished_now])
                    # import ipdb; ipdb.set_trace()  # noqa
                    break

            # Transition to next state:
            reward = reward.at[~finished].add(self.predict_rew(state[~finished], action[~finished]).astype(jnp.float32))
            state = state.at[~finished].set(self.predict_obs(state[~finished], action[~finished]).astype(jnp.float32))
            num_t = num_t.at[~finished].add(1)

        # TODO: ideally, here we should check for alt winconns again
        #       for cases where env finished exactly after MAX_TRANSITIONS
        #       (a very edge case though)

        num_truncated = (~finished).sum()

        if num_truncated > 0:
            # self.num_truncations += num_truncated
            # print(f"WARNING: state still in progress after {self.max_transitions} transitions")
            pass

        # Give one final step reward + a term reward
        # (the rew_model cannot predict terminal rewards)
        #
        # NOTE: excluding initial_finished states i.e. states which finished
        #       in prev dream phases and already have term rewards
        term_this_dream_phase = terminated & (~initial_terminated)
        if term_this_dream_phase.any():
            s = state[term_this_dream_phase]
            reward = reward.at[term_this_dream_phase].add(1000 * (
                s[:, self.index_enemy_value_lost_now_rel] - s[:, self.index_my_value_lost_now_rel]
                + self.reward_dmg_mult * (s[:, self.index_enemy_dmg_received_now_rel] - s[:, self.index_my_dmg_received_now_rel])
                + self.reward_term_mult * (s[:, self.index_enemy_value_lost_acc_rel0] - s[:, self.index_my_value_lost_acc_rel0])
            ))

        return state, reward, terminated, num_t

    @classmethod
    def create_model(cls, **model_kwargs):
        # must be 1st (env changes workdir)
        import torch
        torch_state_obs = torch.load("hauzybxn-model.pt", weights_only=True, map_location="cpu")
        torch_state_rew = torch.load("aexhrgez-model.pt", weights_only=True, map_location="cpu")
        torch_state_act = torch.load("ogyesvkb-model.pt", weights_only=True, map_location="cpu")

        model = Predictor(**model_kwargs)

        params = model.init(
            rngs={"params": jax.random.PRNGKey(0)},
            initial_state=jnp.zeros([1, STATE_SIZE]),
            initial_action=jnp.array([1]),
            method=Predictor.setup_params
        )

        from rl.world.t10n.jax.flax.load_utils import load_params_from_torch_state

        params_obs = load_params_from_torch_state(
            unfreeze(params["params"]["obs_model"]),
            torch_state_obs,
            head_names=["global", "player", "hex"]
        )

        params_rew = load_params_from_torch_state(
            unfreeze(params["params"]["rew_model"]),
            torch_state_rew,
            head_names=["reward"]
        )

        params_act = load_params_from_torch_state(
            unfreeze(params["params"]["act_model"]),
            torch_state_act,
            head_names=["main", "hex"],
            action=False
        )

        params = freeze({
            "params": {
                "obs_model": params_obs,
                "rew_model": params_rew,
                "act_model": params_act,
            }
        })

        return model, params


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

    test_cuda = any(device.platform == 'gpu' for device in jax.devices())

    model, params = Predictor.create_model(
        jit=test_cuda,  # very slow (10-100x slower than torch, even on GPU)
        max_transitions=5,
        side=(env_kwargs["role"] == "defender"),
        reward_dmg_mult=env_kwargs["reward_dmg_mult"],
        reward_term_mult=env_kwargs["reward_term_mult"],
    )

    if test_cuda:
        import torch
        from rl.world.i2a import ImaginationCore
        torch_model = ImaginationCore(
            max_transitions=model.max_transitions,
            side=model.side,
            reward_step_fixed=0,
            reward_dmg_mult=model.reward_dmg_mult,
            reward_term_mult=model.reward_term_mult,
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

    nsteps = 10
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

    if test_cuda:
        print("------- BATCH TEST JAX ON CUDA ----------")

        print("Benchmarking jax (%dx%d)..." % (nsplits, len(split_act[0])))
        batch_start = time.perf_counter()
        for b_obs, b_act in zip(split_obs, split_act):
            state, reward, done, num_t = model.apply(
                params,
                initial_state=jnp.array(b_obs),
                initial_action=jnp.array(b_act)
            )
            print(".", end="", flush=True)
        batch_end = time.perf_counter()
        print("\ntime: %.2fs" % (batch_end - batch_start))

        # cuda_obs = torch.as_tensor(np.array([o["observation"] for o in buffer["obs"]]), device=torch_model.device)
        # cuda_act = torch.as_tensor(buffer["act"], device=torch_model.device, dtype=torch.int64)

        print("------- BATCH TEST TORCH ON CUDA ----------")
        print("Benchmarking cuda (%dx%d)..." % (nsplits, len(split_act[0])))
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

    state, reward, done, num_t = model.apply(
        params,
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

    print("Benchmarking (5)...")
    jax_start = time.perf_counter()
    for _ in range(5):
        model.apply(
            params,
            initial_state=jnp.expand_dims(start_obs, axis=0),
            initial_action=jnp.array([start_act])
        )
        print(".", end="", flush=True)
    jax_end = time.perf_counter()
    print("\ntime: %.2fs" % (jax_end - jax_start))

    # print("Benchmarking jit (5)...")
    # jax_start = time.perf_counter()
    # for _ in range(5):
    #     jit_fwd(params, obs=start_obs, act=start_act, key=key)
    #     print(".", end="", flush=True)
    # jax_end = time.perf_counter()
    # print("\ntime: %.2fs" % (jax_end - jax_start))

    import ipdb; ipdb.set_trace()  # noqa
    pass
