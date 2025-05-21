import re
import torch

from .world import WorldModel
from .t10n import t10n
from .p10n import p10n


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


def render_dream(dream):
    try:
        ary_lines = [prepare(s, a, None, "Dream step %d:" % i) for i, (s, a) in enumerate(dream)]
    except Exception:
        print("!!!!!!!!! ERRROR PREPARING !!!!!. TEMP RENDER:")
        ary_lines = [print("\n").join(prepare(s, a, None, "Dream step %d:" % i)) for i, (s, a) in enumerate(dream)]
        print("\n\n")

    print("")
    print("\n".join([(" → ".join(rowlines)) for rowlines in zip(*ary_lines)]))
    print("")


if __name__ == "__main__":
    dream = []
    wm = WorldModel()

    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    from vcmi_gym.envs.v12.decoder.decoder import Decoder

    env = VcmiEnv(
        mapname="gym/generated/evaluation/8x512.vmap",
        opponent="BattleAI",
        swap_sides=0,
        role="defender",
        # random_heroes=1,
        # random_obstacles=1,
        # town_chance=20,
        # warmachine_chance=30,
        # random_stack_chance=65,
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

    dream = []
    callback = lambda s, a: dream.append((s, a))

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

        def do_dream(t10n_strat, p10n_strat):
            with torch.no_grad():
                wm(t(start_obs), t(start_act), t10n_strat, p10n_strat, callback=callback)

        # Change in pdb to repeat same step
        # (e.g. to see "alternate dreams" when strategy is SAMPLES)
        pdb_state = dict(repeat=False)

        dream.clear()
        do_dream(t10n.Reconstruction.PROBS, p10n.Prediction.PROBS)
        render_dream(dream)

        import ipdb; ipdb.set_trace()  # noqa
        print("")
