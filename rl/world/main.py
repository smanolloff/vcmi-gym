import re
import torch

from .t10n import t10n
from .p10n import p10n

from .util.constants_v11 import (
    GLOBAL_ATTR_MAP
)

IDX_BSAP_START = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][1]
IDX_BSAP_END = GLOBAL_ATTR_MAP["BATTLE_SIDE_ACTIVE_PLAYER"][2] + IDX_BSAP_START

IDX_WINNER_START = GLOBAL_ATTR_MAP["BATTLE_WINNER"][1]
IDX_WINNER_END = GLOBAL_ATTR_MAP["BATTLE_WINNER"][2] + IDX_WINNER_START

# Attacker army: 1 phoenix
# Defender army: 3 arrow towers + 8 stacks (incl. ballista)
# => Transitions:
#   1. Catapult
#   2. 3x arrow tower (wait)
#   3. Phoenix
#   ---- transitions start:
#   4. 8x stacks (wait)
#   5. 8x stacks (act)
#   6. 3x arrow tower (act)
#   7. (new round) Catapult
#   8. 3x arrow tower (wait)
#   ---- transitions end
#   = 23... (the "prediction" will likely be totally wrong)
MAX_TRANSITIONS = 23


class WorldModel:
    def __init__(
        self,
        device=torch.device("cpu"),
        transition_model_file="data/world/t10n/cvftmtsn-model.pt",
        action_prediction_model_file="data/world/p10n/czzklpfu-model.pt",
    ):
        def load_weights(model, file):
            model.load_state_dict(torch.load(file, weights_only=True, map_location=device), strict=True)

        self.transition_model = t10n.TransitionModel(device)
        self.transition_model.eval()

        self.action_prediction_model = p10n.ActionPredictionModel(device)
        self.action_prediction_model.eval()

        load_weights(self.transition_model, transition_model_file)
        load_weights(self.action_prediction_model, action_prediction_model_file)

    def full_transition(self, state, action, t10n_strategy, p10n_strategy):
        with torch.no_grad():
            initial_player = state[:, IDX_BSAP_START:IDX_BSAP_END].argmax(dim=1)
            initial_winner = state[:, IDX_WINNER_START:IDX_WINNER_END].argmax(dim=1)
            # => (B)  # values 0=done, 1=red or 2=blue

            dream = [(state[0].numpy(), action[0].item())]

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

                if t10n_strategy == t10n.Reconstruction.PROBS:
                    # Rendering probs will likely fail => collapse first
                    greedy = self.transition_model.reconstruct(state_logits_in_progress, strategy=t10n.Reconstruction.GREEDY)
                    dream.append((greedy[0].numpy(), action_in_progress.item()))
                else:
                    # Rendering greedy is ok, samples is kind-of-ok => leave as-is
                    dream.append((state_in_progress[0].numpy(), action_in_progress.item()))

            if idx_in_progress.numel() > 0:
                print(f"WARNING: state still in progress after {MAX_TRANSITIONS} transitions")

            # Finally append latest state (it should have no action)
            if t10n_strategy == t10n.Reconstruction.PROBS:
                greedy = self.transition_model.reconstruct(state_logits, strategy=t10n.Reconstruction.GREEDY)
                dream.append((greedy[0].numpy(), -1))
            else:
                dream.append((state[0].numpy(), -1))

            return dream


def prepare(state, action, reward, headline):
    bf = Decoder.decode(state)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    rewtxt = "" if reward is None else "Reward: %s" % round(reward, 2)
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
    render["bf_lines"].append(env.__class__.action_text(action, bf=bf).rjust(render["bf_maxprintlen"]))
    return render["bf_lines"]


if __name__ == "__main__":
    wm = WorldModel()

    from vcmi_gym.envs.v11.vcmi_env import VcmiEnv
    from vcmi_gym.envs.v11.decoder.decoder import Decoder

    env = VcmiEnv(
        mapname="gym/generated/evaluation/8x512.vmap",
        opponent="BattleAI",
        swap_sides=0,
        # random_heroes=1,
        # random_obstacles=1,
        # town_chance=20,
        # warmachine_chance=30,
        # random_terrain_chance=100,
        # tight_formation_chance=30,
        conntype="thread"
    )

    t = lambda x: torch.as_tensor(x).unsqueeze(0)

    env.reset()
    print("Testing accuracy for 1000 steps...")

    matches = []
    losses = []
    episodes = 0
    verbose = False

    def _print(txt):
        if verbose:
            print(txt)

    total_steps = 1000

    for step in range(total_steps):
        if env.terminated or env.truncated:
            env.reset()
            episodes += 1
        act = env.random_action()
        obs0, rew, term, trunc, _info = env.step(act)
        done = term or trunc

        num_transitions = len(obs0["transitions"]["observations"])

        if num_transitions != 4:
            continue

        env.render_transitions(add_regular_render=False)
        print("^ Transitions: %d" % num_transitions)

        start_obs = obs0["transitions"]["observations"][0]
        start_act = obs0["transitions"]["actions"][0]
        print("Dream act: %s" % start_act)

        # Change in pdb to repeat same step
        # (e.g. to see "alternate dreams" when strategy is SAMPLES)
        pdb_state = dict(repeat=False)

        def do_dream(t10n_strategy, p10n_strategy):
            dream = wm.full_transition(t(start_obs), t(start_act), t10n_strategy, p10n_strategy)

            try:
                ary_lines = [prepare(s, a, None, "Dream step %d:" % i) for i, (s, a) in enumerate(dream)]
            except Exception:
                print("!!!!!!!!! ERRROR PREPARING !!!!!. TEMP RENDER:")
                ary_lines = [print("\n").join(prepare(s, a, None, "Dream step %d:" % i)) for i, (s, a) in enumerate(dream)]
                print("\n\n")

            print("")
            print("\n".join([(" → ".join(rowlines)) for rowlines in zip(*ary_lines)]))
            print("")
            return dream

        do_dream(t10n.Reconstruction.PROBS, p10n.Prediction.PROBS)
        import ipdb; ipdb.set_trace()  # noqa
        print("")
