import gymnasium as gym
import time
import random


def benchmark(total_steps):
    env = gym.make("local/VCMI-v0")
    obs, info = env.reset()
    term = False
    trunc = False
    steps = 0
    resets = 0
    tmpsteps = 0
    tmpresets = 0
    tstart = time.time()
    t0 = time.time()
    benchside = info["side"]
    ew = env.unwrapped
    update_every = total_steps // 100  # 100 updates total (every 1%)
    report_every = 10  # every 10%

    assert total_steps % 10 == 0

    print("* Map: %s" % ew.mapname)
    print("* Attacker: %s %s" % (ew.attacker, ew.attacker_model if ew.attacker == "MMAI_MODEL" else ""))
    print("* Defender: %s %s" % (ew.defender, ew.defender_model if ew.defender == "MMAI_MODEL" else ""))
    print("* Total steps: %d" % total_steps)
    print("")

    was_term = False
    two_users = ew.attacker == "MMAI_USER" and ew.defender == "MMAI_USER"

    try:
        while steps < total_steps:
            if term or trunc:
                assert not was_term
                was_term = two_users
                if info["side"] == benchside:
                    resets += 1
                    tmpresets += 1

                obs, info = env.reset()
                term = False
                trunc = False
            elif was_term and two_users:
                # means we just processed 1st env's terminal obs, now 2nd
                was_term = False
                obs, info = env.reset()
            else:
                valid_action_indexes = [i for i, v in enumerate(ew.action_masks()) if v]
                act = random.choice(valid_action_indexes)
                obs, _, term, trunc, info = env.step(act)

            # reset is also a "step" (aka. a round-trip to VCMI)
            steps += 1

            if steps % update_every == 0:
                tmpsteps += update_every
                percentage = (steps / total_steps) * 100
                print("\r%d%%..." % percentage, end="", flush=True)

                if steps % (update_every*report_every) == 0:
                    s = time.time() - t0
                    print(" steps/s: %-6.2f resets/s: %-6.2f" % (tmpsteps/s, tmpresets/s))
                    tmpsteps = 0
                    tmpresets = 0
                    t0 = time.time()
                    # avoid hiding the percent
                    if percentage < 100:
                        print("\r%d%%..." % percentage, end="", flush=True)

        print("")
        print("* Total time: %.2f seconds" % (time.time() - tstart))
        print("* Total steps: %d" % steps)
        print("* Total resets: %d" % resets)
    finally:
        env.close()
