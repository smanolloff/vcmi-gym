import gymnasium as gym
import time
import importlib

from . import common


def load_model(mod_name, cls_name, file):
    print("Loading %s model from %s" % (cls_name, file))
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name).load(file)


def spectate(
    fps,
    reset_delay,
    model_file,
    model_mod,
    model_cls,
):
    model = load_model(model_mod, model_cls, model_file)
    env = gym.make("local/VCMI-v0")

    try:
        while True:
            obs, info = env.reset()
            common.play_model(env, fps, model, obs)
            time.sleep(reset_delay)
    finally:
        env.close()
