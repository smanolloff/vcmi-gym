import re
import os
import torch

from .i2a import ImaginationCore
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
        ary_lines = [prepare(s, a, None, "Dream step %d:" % i) for i, (s, a) in enumerate(dream[1:])]
    except Exception:
        print("!!!!!!!!! ERRROR PREPARING !!!!!. TEMP RENDER:")
        ary_lines = [print("\n").join(prepare(s, a, None, "Dream step %d:" % i)) for i, (s, a) in enumerate(dream)]
        print("\n\n")

    print("")
    print("\n".join([(" → ".join(rowlines)) for rowlines in zip(*ary_lines)]))
    print("")


if __name__ == "__main__":
    dream = []
    dream2 = []

    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv
    from vcmi_gym.envs.v12.decoder.decoder import Decoder

    oldcwd = os.getcwd()
    env = VcmiEnv(
        # mapname="gym/generated/evaluation/8x512.vmap",
        mapname="gym/A1.vmap",
        opponent="StupidAI",
        swap_sides=0,
        role="defender",
        random_heroes=1,
        random_obstacles=1,
        town_chance=20,
        warmachine_chance=30,
        random_stack_chance=65,
        random_terrain_chance=100,
        tight_formation_chance=30,
    )

    assert env.role == "defender"
    ic = ImaginationCore(
        side=1,
        reward_step_fixed=env.reward_cfg.step_fixed,
        reward_dmg_mult=env.reward_cfg.dmg_mult,
        reward_term_mult=env.reward_cfg.term_mult,
        max_transitions=10,
        transition_model_file=f"{oldcwd}/hauzybxn-model.pt",
        action_prediction_model_file=f"{oldcwd}/ogyesvkb-model.pt",
        reward_prediction_model_file=f"{oldcwd}/aexhrgez-model.pt",
    )

    t = lambda x: torch.as_tensor(x).unsqueeze(0)

    env.reset()

    matches = []
    losses = []
    episodes = 0
    verbose = False

    def _print(txt):
        if verbose:
            print(txt)

    dream = []
    callback = lambda s, a: dream.append((s, a))
    callback2 = lambda s, a: dream2.append((s, a))
    rewloss = torch.tensor(0.)
    rewloss2 = torch.tensor(0.)

    total_steps = 500
    step = 0

    print("Testing accuracy for %d steps..." % total_steps)

    while step < total_steps:
        if env.terminated or env.truncated:
            env.reset()
            episodes += 1
        act = env.random_action()
        obs0, rew, term, trunc, _info = env.step(act)
        done = term or trunc

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

        def do_dream(t10n_strat, p10n_strat, callback):
            with torch.no_grad():
                return ic(t(start_obs), t(start_act), t10n_strat, p10n_strat, callback=callback, obs0=obs0, debug=True)

        # Change in pdb to repeat same step
        # (e.g. to see "alternate dreams" when strategy is SAMPLES)
        pdb_state = dict(repeat=False)

        dream.clear()
        state, reward, done, num_t = do_dream(t10n.Reconstruction.GREEDY, p10n.Prediction.GREEDY, callback)
        # render_dream(dream)
        print("Predicted transitions: %s, real: %s" % (num_t, num_transitions-1))
        print("[GREEDY] Done: %s" % str([done, done.item()]))
        print("[GREEDY] Reward: %s" % str([rew, reward.item()]))
        print("[GREEDY] Reward loss (done=%s): %.2f" % (done[0].long().item(), torch.nn.functional.mse_loss(torch.tensor(rew), reward[0])))

        # env.render_transitions()

        # dream2.clear()
        # state2, reward2, done2 = do_dream(t10n.Reconstruction.GREEDY, p10n.Prediction.PROBS, callback2)
        # # render_dream(dream2)
        # # print("[PROBS] Done: %s" % str([done2, done2.item()]))
        # print("[PROBS] Reward: %s" % str([rew, reward2.item()]))
        # print("[PROBS] Reward loss (done=%s): %.2f" % (done2[0].long().item(), torch.nn.functional.mse_loss(torch.tensor(rew), reward2[0])))

        # rl = torch.nn.functional.mse_loss(torch.tensor(rew), reward[0])
        # rl2 = torch.nn.functional.mse_loss(torch.tensor(rew), reward2[0])
        # rewloss += rl
        # rewloss2 += rl2

        # print("[PROBS-GREEDY] diff: %.2f (done=%s)" % ((rl2 - rl), done))

        step += 1

    print("[GREEDY] Mean reward loss: %.2f" % (rewloss / total_steps))
    print("[PROBS] Mean reward loss: %.2f" % (rewloss2 / total_steps))
    print("[PROBS-GREEDY] loss: %.2f" % ((rewloss2 / total_steps) - (rewloss / total_steps)))
