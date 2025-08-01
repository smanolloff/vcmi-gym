import argparse
import torch

from . import vae
from .weights import build_feature_weights
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
            random_stack_chance=70,
            tight_formation_chance=30,
        )
        do_test(model, env)


def load_for_test(file):
    model = vae.VAE(deterministic=True)
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

        for obs_real in obs["transitions"]["observations"]:
            obs_decoded = model(torch.as_tensor(obs_real).unsqueeze(0))[0]
            obs_recon = model.decoder.reconstruct(obs_decoded)

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

            lines_real = prepare(obs_real, -1, None, "Real:")
            lines_recon = prepare(obs_recon, -1, None, "Reconstructed:")

            total_loss, losses = vae.compute_decode_losses(
                logger=None,
                abs_index=model.abs_index,
                loss_weights=weights,
                real_obs=torch.as_tensor(obs_real).unsqueeze(0),
                decoded_obs=obs_decoded.unsqueeze(0),
            )

            # print("Losses | Obs: binary=%.4f, cont=%.4f, categorical=%.4f, threshold=%.4f" % losses)
            print("Losses: %s | %s" % (total_loss, losses))

            # print(Decoder.decode(obs_prev).render(0))
            # for i in range(len(bfields)):
            print("")
            print("\n".join([(" ".join(rowlines)) for rowlines in zip(lines_real, lines_recon)]))
            print("")

            import ipdb; ipdb.set_trace()  # noqa


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
            model_creator=vae.VAE,
            weights_builder=build_feature_weights,
            buffer_creator=vae.Buffer,
            vcmi_dataloader_functor=vae.vcmi_dataloader_functor,
            s3_dataloader_functor=None,
            eval_model_fn=vae.eval_model,
            train_model_fn=vae.train_model
        )
        if args.action == "train":
            train(**dict(common_args, sample_only=False))
        elif args.action == "sample":
            train(**dict(common_args, sample_only=True))
