from vcmi_gym.tools.main import run

import sys
import os
import numpy as np
import ast
import wandb


if __name__ == "__main__":
    seed = int(np.random.default_rng().integers(2**31))
    run_id = os.environ["WANDB_RUN_ID"]

    # Fetch and expand `out_dir_template`
    cfgstr = next(v[4:] for v in sys.argv if v.startswith("--c="))
    cfg = ast.literal_eval(cfgstr)
    out_dir = cfg["out_dir_template"].format(run_id=run_id, seed=seed)
    out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    print("out dir: %s" % out_dir)

    # 4: Patch tensorboard before wandb.init
    # (to suppresses wandb warnings from adversarial training)
    # DISABLED: it makes lognames from pure PPO training too long
    # wandb.tensorboard.patch(root_logdir=out_dir)
    wandb.init(sync_tensorboard=False)

    action = wandb.config["action"]
    config = dict(
        wandb.config["c"],
        run_id=run_id,
        seed=seed,
        out_dir=out_dir,
    )

    run(action, config)
    wandb.save(f"{out_dir}/model.zip", base_path=out_dir)
    wandb.save(f"{out_dir}/metadata.yml", base_path=out_dir)
    wandb.finish()
