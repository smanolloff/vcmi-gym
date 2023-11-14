import gymnasium as gym
import time

from vcmi_gym import VcmiEnv


def benchmark(steps):
    env = gym.make("local/VCMI-v0")
    env.reset()
    resets = 0

    try:
        env.reset()
        time_start = time.time()

        action = 2 + 75*8 + 1
        for i in range(steps):
            obs, rew, term, trunc, info = env.step(action)
            action += 1

            if term:
                action = 2 + 75*8 + 1
                env.reset()
                resets += 1

            if i % 1000 == 0:
                percentage = (i / steps) * 100
                print("\r%d%%..." % percentage, end="", flush=True)

        seconds = time.time() - time_start
        sps = steps / seconds
        print("\n\n%.2f steps/s (%s steps in %.2f seconds, %d resets total)" % (sps, steps, seconds, resets))
    finally:
        env.close()
