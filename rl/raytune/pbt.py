import copy
import ray.tune
import ast
from dataclasses import asdict
import datetime
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


def update_config_value(cfg, path, value):
    keys = path.split('.')
    d = cfg

    # Traverse the dict to find the position of the final key
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]

    old_value = d.get(keys[-1])
    new_value = ast.literal_eval(value)
    action = "No change for" if old_value == new_value else "Overwrite"
    print("%s %s: %s -> %s" % (action, path, old_value, value))
    d[keys[-1]] = new_value
    return old_value, new_value


def extract_initial_hyperparams(runtime_cfg, hyperparam_mutations, res=None):
    if res is None:
        res = {}
    for k in runtime_cfg:
        if k in hyperparam_mutations:
            if isinstance(runtime_cfg[k], dict):
                assert isinstance(hyperparam_mutations[k], dict), "not a dict: hyperparam_mutations[%s]" % k
                res[k] = extract_initial_hyperparams(runtime_cfg[k], hyperparam_mutations[k])
            else:
                assert isinstance(hyperparam_mutations[k], list), "not a list: allowed_keys_dict[%s]" % k
                assert isinstance(runtime_cfg[k], (int, float, str)), "not an int/float/str: src_dict[%s]" % k
                k not in res, "'%s' key already present in res" % k
                res[k] = runtime_cfg[k]
    return res


def main(alg, exp_name, resume_path, config_overrides=[]):
    cfg = copy.deepcopy(config)

    if resume_path:
        import torch
        import wandb
        agent = torch.load(resume_path)
        config_overrides.insert(0, f"agent_load_file={repr(resume_path)}")
        run = wandb.Api().run(f"s-manolloff/vcmi-gym/{agent.args.run_id}")
        cfg = copy.deepcopy(run.config)
        alg = cfg["_raytune"]["algo"]
        exp_name = cfg["_raytune"]["experiment_name"]

        # Special key holding the ORIGINAL run_id to resume. Do not remove.
        cfg["_raytune"]["resumed_run_id"] = agent.args.run_id

        cfg["_raytune"]["initial_hyperparams"] = extract_initial_hyperparams(
            asdict(agent.args),
            cfg["_raytune"]["hyperparam_mutations"]
        )

        print("Using initial hyperparams: %s" % cfg["_raytune"]["initial_hyperparams"])

        cfg["_raytune"]["resumes"] = cfg["_raytune"].get("resumes", [])

        resume = {
            "resumed_at": datetime.datetime.now().astimezone().isoformat(),
            "prev_run_id": agent.args.run_id,
            "prev_trial_id": getattr(agent.args, "trial_id", "(no trial_id)"),
            "overrides": {}
        }

        # config_overrides is a list of "path.to.key=value"
        for co in config_overrides:
            name, value = co.split("=")
            oldvalue, newvalue = update_config_value(cfg, name, value)
            if oldvalue != newvalue:
                resume["overrides"][name] = [oldvalue, newvalue]

        cfg["_raytune"]["resumes"].append(resume)

    mutations = cfg["_raytune"]["hyperparam_mutations"]

    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    scheduler = ray.tune.schedulers.PopulationBasedTraining(
        time_attr=cfg["_raytune"]["time_attr"],
        metric=cfg["_raytune"]["metric"],
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
