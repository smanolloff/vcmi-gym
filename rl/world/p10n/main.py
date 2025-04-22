import argparse
import torch

from . import p10n
from ..util.train import train
from ..util.misc import safe_mean

from ..util.constants_v11 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


def test(weights_file):
    from vcmi_gym.envs.v11.vcmi_env import VcmiEnv

    with torch.no_grad():
        model = load_for_test(weights_file)
        model.eval()
        # env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap", conntype="thread", random_heroes=1, swap_sides=1)
        env = VcmiEnv(
            mapname="gym/generated/evaluation/8x512.vmap",
            opponent="BattleAI",
            swap_sides=0,
            random_heroes=1,
            random_obstacles=1,
            town_chance=20,
            warmachine_chance=30,
            random_terrain_chance=100,
            tight_formation_chance=30,
            conntype="thread"
        )
        do_test(model, env)


def load_for_test(file):
    model = p10n.ActionPredictionModel()
    model.eval()
    print(f"Loading {file}")
    weights = torch.load(file, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(weights, strict=True)
    return model


def do_test(model, env):
    from vcmi_gym.envs.v11.decoder.decoder import Decoder

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
        for i in range(1, num_transitions):
            obs = obs0["transitions"]["observations"][i]
            act = obs0["transitions"]["actions"][i]

            if act == -1:
                # Usually, act=-1 on every *last* transition
                # We care about it only if it's the last for the entire episode
                assert i == num_transitions - 1, f"{i} == {num_transitions} - 1"
                if not done:
                    continue

            logits_main, logits_hex = model(t(obs))
            probs = model.predict_(t(obs), logits=(logits_main, logits_hex), strategy=p10n.Prediction.PROBS)[0]

            # act_pred = torch.multinomial(probs, num_samples=1)
            act_pred = probs.argmax()

            losses.append(p10n.compute_loss(t(act), logits_main, logits_hex))
            matches.append(act == act_pred)

            if not verbose:
                continue

            bf = Decoder.decode(obs)
            _print(bf.render(0))

            probs_main = logits_main[0].softmax(0)
            probs_hex = logits_hex[0, :, 0].softmax(0)

            # Probs + indices for top 3 hexes
            k = 3
            topk_hexes = probs_hex.topk(k)
            # => (k,)

            # Probs + indices for all 14 actions of the top 3 hexes
            probs_actions_topk_hexes = logits_hex[0, topk_hexes.indices, 1:].softmax(1)
            # => (k, N_HEX_ACTIONS)

            topk_hexes_topk_actions = probs_actions_topk_hexes.topk(k, dim=1)
            # => (k, k)  # dim0=hexes, dim1=actions

            hexactnames = list(p10n.HEX_ACT_MAP.keys())

            _print("Real action: %d (%s)" % (act, env.action_text(act, bf=bf)))
            _print("Pred action: %d (%s) %s" % (act_pred, env.action_text(act_pred, bf=bf), "✅" if act == act_pred else "❌"))

            losses = p10n.compute_loss(t(act), logits_main, logits_hex)
            _print("Total loss: %.3f (%.2f + %.2f + %.2f)" % (sum(losses), *losses))
            _print("Probs:")

            assert p10n.MainAction.HEX == len(p10n.MainAction) - 1
            for ma in p10n.MainAction:
                _print(" * %s: %.2f" % (ma.name, probs_main[ma.value]))

            for i in range(k):
                hex = bf.get_hex(topk_hexes.indices[i].item())
                hex_desc = "Hex(y=%s x=%s)/%.2f" % (hex.Y_COORD.v, hex.X_COORD.v, topk_hexes.values[i])
                hexact_desc = []
                for ind, prob in zip(topk_hexes_topk_actions.indices[i], topk_hexes_topk_actions.values[i]):
                    actcalc = 2 + topk_hexes.indices[i]*len(p10n.HEX_ACT_MAP) + ind
                    text = "%d: %s/%.2f" % (actcalc, hexactnames[ind.item()], prob.item())
                    hexact_desc.append(text.ljust(23))
                _print("    - [k=%d] %-18s => %s" % (i, hex_desc, " ".join(hexact_desc)))

            # import ipdb; ipdb.set_trace()  # noqa
            _print("==========================================================")

        if step % (total_steps // 10) == 0:
            print("(progress: %.0f%%)" % (100 * (step / total_steps)))

    print("Episodes: %d" % episodes)
    print("Loss: %s" % (safe_mean(losses)))
    print("Accuracy: %.2f%%" % (100 * (sum(matches) / len(matches))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="FILE", help="config file to resume training or model file to test")
    parser.add_argument("--dry-run", action="store_true", help="do not save anything to disk (implies --no-wandb)")
    parser.add_argument("--no-wandb", action="store_true", help="do not initialize wandb")
    parser.add_argument("--loglevel", metavar="LOGLEVEL", default="INFO", help="DEBUG | INFO | WARN | ERROR")
    parser.add_argument('action', metavar="ACTION", type=str, help="train | test | sample")
    args = parser.parse_args()

    if args.dry_run:
        args.no_wandb = True

    if args.action == "test":
        test(args.f)
    else:
        from .config import config
        common_args = dict(
            config=config,
            resume_config=args.f,
            loglevel=args.loglevel,
            dry_run=args.dry_run,
            no_wandb=args.no_wandb,
            # sample_only=False,
            model_creator=p10n.ActionPredictionModel,
            buffer_creator=p10n.Buffer,
            vcmi_dataloader_functor=p10n.vcmi_dataloader_functor,
            s3_dataloader_functor=None,
            eval_model_fn=p10n.eval_model,
            train_model_fn=p10n.train_model
        )
        if args.action == "train":
            train(**dict(common_args, sample_only=False))
        elif args.action == "sample":
            train(**dict(common_args, sample_only=True))
