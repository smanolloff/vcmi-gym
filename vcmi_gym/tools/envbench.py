import sys
import os
import time
import gymnasium as gym

from vcmi_gym.envs.util.wrappers import LegacyObservationSpaceWrapper

from types import SimpleNamespace
from vcmi_gym.envs.v12 import VcmiEnv


def bench_plain_env(env_kwargs, total_steps):
    print("Benchmarking PLAIN env...")
    env = VcmiEnv(**env_kwargs)
    term = False
    trunc = False

    for _ in range(total_steps):
        if term or trunc:
            env.reset()
            term = False
            trunc = False
        obs, rew, term, trunc, info = env.step(env.random_action())


def bench_vector_env(env_kwargs, num_envs, total_steps):
    print("Benchmarking ASYNC(%d) venv..." % num_envs)
    pid = os.getpid()
    dummy_env = SimpleNamespace(
        metadata={'render_modes': ['ansi', 'rgb_array'], 'render_fps': 30},
        render_mode='ansi',
        action_space=VcmiEnv.ACTION_SPACE,
        observation_space=VcmiEnv.OBSERVATION_SPACE["observation"],
        close=lambda: None
    )

    env_creator = lambda: dummy_env if os.getpid() == pid else LegacyObservationSpaceWrapper(VcmiEnv(**env_kwargs))
    funcs = [env_creator for _ in range(num_envs)]
    venv = gym.vector.AsyncVectorEnv(funcs, daemon=True, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    for _ in range(total_steps):
        venv.step(venv.call("random_action"))


def main():
    num_envs = int(sys.argv[1])
    total_vsteps = 1000

    env_kwargs = dict(
        mapname="gym/generated/4096/4096-6stack-100K-01.vmap",
        random_heroes=0,
        random_obstacles=0,
        opponent="StupidAI",
        max_steps=1000,
        # swap_sides=1,
    )

    print("Testing %d envs for %d vsteps..." % (num_envs, total_vsteps))
    t = time.time()
    if num_envs == 0:
        bench_plain_env(env_kwargs, total_vsteps)
    else:
        bench_vector_env(env_kwargs, num_envs, total_vsteps)

    elapsed = time.time() - t
    print("Steps: %d" % total_vsteps)
    print("num_envs: %d" % num_envs)
    print("Elapsed: %.2fs" % elapsed)
    print("steps/s: %d" % ((total_vsteps * max(num_envs, 1)) / elapsed))


if __name__ == "__main__":
    main()
