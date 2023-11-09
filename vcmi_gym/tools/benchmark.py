import gymnasium as gym
import time

from vcmi_gym import VcmiEnv


def benchmark(steps):
    # env = gym.make("local/VCMI-v0")
    env = VcmiEnv(mapname="pikemen.vmap", vcmi_loglevel="error")
    action = 2 + 75*8 + 1

    try:
        env.reset()
        time_start = time.time()

        for i in range(steps):
            obs, rew, term, trunc, info = env.step(action)
            # obs, rew, term, trunc, info = env.step(0)
            print("======== obs: (hidden)")
            print("======== rew: %s" % rew)
            print("======== term: %s" % term)
            print("======== trunc: %s" % trunc)
            print("======== info: %s" % info)

            if term:
                breakpoint()
                env.reset()
                action = 2 + 75*8 + 1

            action += 1

            if i % 1000 == 0:
                percentage = (i / steps) * 100
                print("\r%d%%..." % percentage, end="", flush=True)

        seconds = time.time() - time_start
        sps = steps / seconds
        print("\n\n%.2f steps/s (%s steps in %.2f seconds)" % (sps, steps, seconds))
    finally:
        env.close()
