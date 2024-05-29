import copy
import ray.tune
from . import common
from .pbt_config import config


def convert_to_param_space(mutations):
    res = {}
    for k, v in mutations.items():
        if isinstance(v, dict):
            res[k] = convert_to_param_space(v)
        elif isinstance(v, list):
            res[k] = ray.tune.choice(v)
        else:
            assert isinstance(v, ray.tune.search.sample.Domain)
    return res


def main(alg, exp_name, resume_path):
    cfg = copy.deepcopy(config)

    if resume_path:
        import torch
        import wandb
        agent = torch.load(resume_path)
        run = wandb.Api().run(f"s-manolloff/vcmi-gym/{agent.args.run_id}")
        cfg = copy.deepcopy(run.config)
        cfg["agent_load_file"] = resume_path
        alg = cfg["_raytune"]["algo"]
        exp_name = cfg["_raytune"]["experiment_name"]

    mutations = cfg["_raytune"]["hyperparam_mutations"]

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    scheduler = ray.tune.schedulers.PopulationBasedTraining(
        time_attr=cfg["_raytune"]["time_attr"],
        metric="rew_mean",
        mode="max",
        perturbation_interval=cfg["_raytune"]["perturbation_interval"],
        hyperparam_mutations=mutations,
        quantile_fraction=cfg["_raytune"]["quantile_fraction"],
        log_config=False,  # used for reconstructing the cfg schedule
        require_attrs=True,
        synch=cfg["_raytune"]["synch"],
    )

    # XXX: "initial" values have a lot of caveats, see note in pbt_config.py
    if cfg["_raytune"]["initial_hyperparams"]:
        initial_cfg = common.common_dict(copy.deepcopy(mutations), cfg["_raytune"]["initial_hyperparams"])
        searcher = ray.tune.search.BasicVariantGenerator(points_to_evaluate=[initial_cfg])
        param_space = convert_to_param_space(mutations)
        tuner = common.new_tuner(alg, exp_name, cfg, scheduler, searcher, param_space)
    else:
        tuner = common.new_tuner(alg, exp_name, cfg, scheduler)

    tuner.fit()
