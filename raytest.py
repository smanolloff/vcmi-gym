import time
import random
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
from ray import tune, train
from ray.tune.schedulers.pb2 import PB2


#
# Minimal test script for demonstrating ray's forceful kill
# of Trainables during cleanup() when SIGINT/SIGTERM is caught
#


def debuglog(func):
    def wrapper(*args, **kwargs):
        args[0].log("Start: %s (args=%s, kwargs=%s)" % (func.__name__, args[1:], kwargs))
        result = func(*args, **kwargs)
        args[0].log("End: %s (return %s)" % (func.__name__, result))
        return result

    return wrapper


class TestTrainer(tune.Trainable):
    def log(self, msg):
        print("[%s] %s" % (self.trial_name, msg))

    @debuglog
    def cleanup(self):
        for i in range(100):
            self.log("cleanup (%d)" % i)
            time.sleep(0.1)

    @debuglog
    def reset_config(self, cfg):
        return True

    @debuglog
    def setup(self, cfg):
        pass

    @debuglog
    def save_checkpoint(self, checkpoint_dir):
        pass

    @debuglog
    def load_checkpoint(self, checkpoint_dir):
        pass

    @debuglog
    def step(self):
        m = random.randint(0, 10)
        # m = 9 # if self.trial_name.endswith("0") else 5
        self.log("simulate some work (%d)..." % m)
        time.sleep(5)
        return {"m": m}


if __name__ == "__main__":
    pb2 = PB2(
        time_attr="training_iteration",
        metric="m",
        mode="max",
        perturbation_interval=2,
        hyperparam_bounds={"p": [0, 1]},
        synch=True  # XXX: time_attr must be training_iteration!!!
    )

    run_config = train.RunConfig(stop={"m": 11})
    tune_config = tune.TuneConfig(scheduler=pb2, reuse_actors=True, num_samples=2)
    tuner = tune.Tuner(
        TestTrainer,
        run_config=run_config,
        tune_config=tune_config,
        param_space={"p": tune.uniform(0, 1)},
    )

    tuner.fit()
