import argparse
import torch

from . import t10n
from ..util.weights import build_feature_weights
from ..util.train import train


def test(weights_file):
    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv

    with torch.no_grad():
        model = load_for_test(weights_file)
        model.eval()
        # env = VcmiEnv(mapname="gym/generated/4096/4x1024.vmap", random_heroes=1, swap_sides=1)
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
        )
        do_test(model, env)


def load_for_test(file):
    model = t10n.TransitionModel()
    model.eval()
    print(f"Loading {file}")
    weights = torch.load(file, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(weights, strict=True)
    return model


def do_test(model, env):
    from vcmi_gym.envs.v12.decoder.decoder import Decoder
    from .config import config

    env.reset()

    weights = build_feature_weights(model, config["weights"])

    for _ in range(10):
        print("=" * 100)
        if env.terminated or env.truncated:
            env.reset()
        act = env.random_action()
        obs, rew, term, trunc, _info = env.step(act)

        # [(obs, act, real_obs), (obs, act, real_obs), ...]
        dream = [(obs["transitions"]["observations"][0], obs["transitions"]["actions"][0], None)]

        for i in range(1, len(obs["transitions"]["observations"])):
            obs_prev = obs["transitions"]["observations"][i-1]
            act_prev = obs["transitions"]["actions"][i-1]
            obs_next = obs["transitions"]["observations"][i]
            # mask_next = obs["transitions"]["action_masks"][i]
            # rew_next = obs["transitions"]["rewards"][i]
            # done_next = (term or trunc) and i == len(obs["transitions"]["observations"]) - 1

            obs_pred_raw = model(torch.as_tensor(obs_prev).unsqueeze(0), torch.as_tensor(act_prev).unsqueeze(0))
            obs_pred_raw = obs_pred_raw[0]
            obs_pred = model.predict(obs_prev, act_prev)
            dream.append((model.predict(*dream[i-1][:2]), obs["transitions"]["actions"][i], obs_next))

            def prepare(state, action, reward, headline):
                import re
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

            lines_prev = prepare(obs_prev, act_prev, None, "Start:")
            lines_real = prepare(obs_next, -1, None, "Real:")
            lines_pred = prepare(obs_pred, -1, None, "Predicted:")

            total_loss, losses = t10n.compute_losses(
                logger=None,
                abs_index=model.abs_index,
                loss_weights=weights,
                next_obs=torch.as_tensor(obs_next).unsqueeze(0),
                pred_obs=obs_pred_raw.unsqueeze(0),
            )

            # print("Losses | Obs: binary=%.4f, cont=%.4f, categorical=%.4f, threshold=%.4f" % losses)
            print("Losses: %s | %s" % (total_loss, losses))

            # print(Decoder.decode(obs_prev).render(0))
            # for i in range(len(bfields)):
            print("")
            print("\n".join([(" ".join(rowlines)) for rowlines in zip(lines_prev, lines_real, lines_pred)]))
            print("")

            # bf_next = Decoder.decode(obs_next)
            # bf_pred = Decoder.decode(obs_pred)

            # print(env.render_transitions())
            # print("Predicted:")
            # print(bf_pred.render(0))
            # print("Real:")
            # print(bf_next.render(0))

            # hex20_pred.stack.QUEUE.raw

            # def action_str(obs, a):
            #     if a > 1:
            #         bf = Decoder.decode(obs)
            #         hex = bf.get_hex((a - 2) // len(HEX_ACT_MAP))
            #         act = list(HEX_ACT_MAP)[(a - 2) % len(HEX_ACT_MAP)]
            #         return "%s (y=%s x=%s)" % (act, hex.Y_COORD.v, hex.X_COORD.v)
            #     else:
            #         assert a == 1
            #         return "Wait"

        # if len(dream) > 2:
        #     print(" ******** SEQUENCE: ********** ")
        #     print(env.render_transitions(add_regular_render=False))
        #     print(" ******** DREAM: ********** ")
        #     rcfg = env.reward_cfg._replace(step_fixed=0)
        #     for i, (obs, act, obs_real) in enumerate(dream):
        #         print("*" * 10)
        #         if i == 0:
        #             print("Start:")
        #             print(Decoder.decode(obs).render(act))
        #         else:
        #             bf_real = Decoder.decode(obs_real)
        #             bf = Decoder.decode(obs)
        #             print(f"Real step #{i}:")
        #             print(bf_real.render(act))
        #             print("")
        #             print(f"Dream step #{i}:")
        #             print(bf.render(act))
        #             print(f"Real / Dream rewards: {env.calc_reward(0, bf_real, rcfg)} / {env.calc_reward(0, bf, rcfg)}:")

    # print(env.render_transitions())

    # print("Pred:")
    # print(Decoder.decode(obs_pred))
    # print("Real:")
    # print(Decoder.decode(obs_real))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar="FILE", help="config file to resume or test")
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
            model_creator=t10n.TransitionModel,
            buffer_creator=t10n.Buffer,
            vcmi_dataloader_functor=t10n.vcmi_dataloader_functor,
            s3_dataloader_functor=None,
            eval_model_fn=t10n.eval_model,
            train_model_fn=t10n.train_model
        )
        if args.action == "train":
            train(**dict(common_args, sample_only=False))
        elif args.action == "sample":
            train(**dict(common_args, sample_only=True))
