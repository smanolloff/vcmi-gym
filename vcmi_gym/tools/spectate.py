import gymnasium as gym
import time
import importlib

from . import common


def load_model(mod_name, cls_name, file):
    print("Loading %s model from %s" % (cls_name, file))
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name).load(file)


def play_model(env, fps, model, obs):
    terminated = False
    clock = common.Clock(fps)
    last_errors = 0

    while not terminated:
        if model.__class__.__name__ == "MaskablePPO":
            action, _states = model.predict(obs, action_masks=env.unwrapped.action_masks())
        else:
            action, _states = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

        if info.get("errors", 0) == last_errors:
            clock.tick()

        last_errors = info.get("errors", 0)

    clock.tick()


def spectate(
    fps,
    reset_delay,
    mapname,
    model_file,
    model_mod,
    model_cls,
):
    envid = "local/VCMI-v0"

    print(f"""
*** NOTE ***

Consider using the *REAL* spectator experience by running

    vcmi_gym/envs/v0/vcmi/rel/bin/mytest {mapname} MMAI {model_file}

""")
    time.sleep(2)

    env = gym.make(envid, mapname=mapname)
    model = load_model(model_mod, model_cls, model_file)

    try:
        while True:
            obs, info = env.reset()
            play_model(env, fps, model, obs)
            time.sleep(reset_delay)
    finally:
        env.close()
