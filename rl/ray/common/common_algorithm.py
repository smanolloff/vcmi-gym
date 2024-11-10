import re
import os
import datetime
import pygit2
import time
import tempfile
import threading
import wandb
from dataclasses import dataclass

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.result import TRIAL_INFO

from . import common_logger
from . import util


class EnvRunnerKeepalive:
    def __init__(self, runner_group, interval, logger):
        self.runner_group = runner_group
        self.interval = interval
        self.logger = logger

    def __enter__(self):
        self.logger.info("Starting env runner keepalive loop")
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        # self.thread = threading.Thread(target=self._debug_loop, daemon=True)
        self.thread.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.info("Stopping env runner keepalive loop...")
        self.event.set()
        self.thread.join()
        self.logger.debug("Stopped env runner keepalive loop")

    def _loop(self):
        while not self.event.is_set():
            self.event.wait(timeout=self.interval)
            self.runner_group.foreach_worker(
                func=lambda w: w.ping(),
                timeout_seconds=10,  # should be instant
                local_env_runner=False,
            )

    def _debug_loop(self):
        def debug(w):
            import ipdb; ipdb.set_trace()  # noqa
        self.runner_group.foreach_worker(func=debug)


@dataclass
class Namespace:
    master_config: dict
    log_interval: int
    run_id: str = None
    run_name: str = None


def init(algo, *args, **kwargs):
    util.silence_log_noise()


def wandb_init(algo):
    run_id = util.gen_id() if algo.trial_id == "default" else algo.trial_id
    print("W&B Run ID is %s (project: %s)" % (run_id, algo.config.user_config.wandb_project))

    run_name = algo.trial_name
    if algo.trial_name == "default":
        run_name = f"{datetime.datetime.now().isoformat()}-debug-{run_id}"
    else:
        run_name = "T%d" % int(algo.trial_name.split("_")[-1])

    algo.ns.run_id = run_id
    algo.ns.run_name = run_name

    if algo.config.user_config.wandb_project:
        wandb.init(
            project=algo.config.user_config.wandb_project,
            group=algo.config.user_config.experiment_name,
            id=util.gen_id() if algo.ns.run_id == "default" else algo.ns.run_id,
            name=algo.ns.run_name,
            resume="allow",
            reinit=True,
            allow_val_change=True,
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
            config=algo.ns.master_config,
            sync_tensorboard=False,
        )

        wandb_log_git_diff(algo)

        # For wandb.log, commit=True by default
        # for wandb_log, commit=False by default
        def wandb_log(*args, **kwargs):
            wandb.log(*args, **dict({"commit": False}, **kwargs))
    else:
        def wandb_log(*args, **kwargs):
            print("*** WANDB LOG AT %s: %s %s" % (datetime.datetime.now().isoformat(), args, kwargs))

    algo.wandb_log = wandb_log


def wandb_add_watch(algo):
    if not wandb.run:
        return

    # XXX: wandb.watch() caused issues during serialization in oldray scripts
    #      (it pollutes the model with non-serializable callbacks)
    #      It seems ray's checkpointing works OK with it
    assert algo.learner_group.is_local
    algo.learner_group.foreach_learner(lambda l: wandb.watch(
        l.module[DEFAULT_POLICY_ID].encoder.encoder,
        log="all",
        log_graph=True,
        log_freq=1000
    ))


def wandb_log_hyperparams(algo):
    to_log = util.common_dict(
        algo.config.user_config.hyperparam_mutations,
        algo.config.to_dict(),
        strict=True
    )

    if to_log:
        algo.logger.info(f"Hyperparam values: {wandb.util.json_dumps_safer(to_log)}")
        algo.wandb_log({f"params/{k}": v for k, v in util.flatten_dict(to_log).items()})


def is_golden_trial(algo):
    return re.match(r"^.+_0+$", algo.trial_id)


# XXX: There's already a wandb "code saving" profile setting
#      See https://docs.wandb.ai/guides/track/log/
#      but it's not working
#      (maybe it requires specifying a directory containing .git)
def wandb_log_git_diff(algo):
    if algo.iteration > 0 or not is_golden_trial(algo):
        return

    git = pygit2.Repository(os.path.dirname(__file__))
    head = str(git.head.target)
    assert head == algo.config.user_config.git_head

    art = wandb.Artifact(name="git-diff", type="text")
    art.description = f"Git diff for HEAD@{head} from {time.ctime(time.time())}"

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
        art.metadata["head"] = head
        temp_file.write(git.diff().patch)
        temp_file.flush()
        art.add_file(temp_file.name, name="diff.patch")
        wandb.run.log_artifact(art)
        # no need to wait for upload (log_artifact creates a local copy)
        # art.wait()


def save_checkpoint(algo, checkpoint_dir):
    learner = algo.learner_group.get_checkpointable_components()[0][1]

    rl_module = learner.module[DEFAULT_MODULE_ID]
    model_file = os.path.join(checkpoint_dir, "jit-model.pt")
    rl_module.jsave(model_file)

    config_file = os.path.join(checkpoint_dir, "mppo_config.json")

    # TRIAL_INFO key contains non-serializable (by wandb) values
    savecfg = {k: v for k, v in algo.config.to_dict().items() if k != TRIAL_INFO}
    with open(config_file, "w") as f:
        f.write(wandb.util.json_dumps_safer(savecfg))

    if wandb.run and is_golden_trial(algo):
        art = wandb.Artifact(name="model", type="model")
        art.description = f"Snapshot of model from {time.ctime(time.time())}"
        art.ttl = datetime.timedelta(days=7)
        art.metadata["step"] = wandb.run.step
        art.add_dir(checkpoint_dir)
        wandb.run.log_artifact(art)


def maybe_load_learner_group(algo):
    if not algo.config.user_config.checkpoint_load_dir:
        return

    path = util.to_abspath("%s/learner_group" % algo.config.user_config.checkpoint_load_dir)
    algo.logger.warning(f"Loading learner group from {path}")
    algo.learner_group.restore_from_path(path)
    broadcast_weights(algo)


# XXX: this works for PPO, but do other algos use more policies?
def maybe_load_model(algo):
    if not algo.config.user_config.model_load_file:
        return

    mapping = algo.config.user_config.model_load_mapping
    path = util.to_abspath(algo.config.user_config.model_load_path)
    algo.logger.warning(f"Loading learner model from {path}")
    algo.learner_group.foreach_learner(lambda l: l.module[DEFAULT_POLICY_ID].jload(path, mapping))
    broadcast_weights(algo)


def broadcast_weights(algo):
    opts = dict(from_worker_or_learner_group=algo.learner_group, inference_only=True)

    algo.logger.info("Broadcasting learner weights to env runners")
    algo.env_runner_group.sync_weights(**opts)

    if algo.eval_env_runner_group:
        algo.logger.info("Broadcasting learner weights to eval env runners")
        algo.eval_env_runner_group.sync_weights(**opts)
