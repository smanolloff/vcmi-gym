import gymnasium as gym
import time


def benchmark(total_steps, actions):
    env = gym.make("local/VCMI-v0")
    env.reset()
    resets = 0
    steps = 0
    time_start = time.time()
    iactions = iter(actions)

    try:
        while steps < total_steps:
            action = next(iactions) - 1
            obs, rew, term, trunc, info = env.step(action)

            if term:
                assert next(iactions, None) is None, "expected no more actions"
                iactions = iter(actions)
                env.reset()
                resets += 1

            if steps % 1000 == 0:
                percentage = (steps / total_steps) * 100
                print("\r%d%%..." % percentage, end="", flush=True)

            steps += 1

        seconds = time.time() - time_start
        sps = steps / seconds
        print("\n\n%.2f steps/s (%s steps in %.2f seconds, %d resets total)" % (sps, steps, seconds, resets))
    finally:
        env.close()
