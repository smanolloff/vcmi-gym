import argparse

from . import p10n
from ..util.train import train

from ..util.constants_v11 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


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
        p10n.test(args.f)
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
