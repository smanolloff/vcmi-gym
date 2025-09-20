import time
import numpy as np

from vcmi_gym.envs.v13.vcmi_env import VcmiEnv, LINK_TYPES


if __name__ == "__main__":
    env = VcmiEnv(
        "gym/generated/4096/4x1024.vmap",
        random_heroes=1,
        random_obstacles=1,
        random_terrain_chance=100,
        tight_formation_chance=10,
        town_chance=10,
        # random_stack_chance=20,  # makes armies unbalanced
        warmachine_chance=40,
    )

    total_steps = 10000
    term = False
    trunc = False
    steps = 0
    resets = 0

    tmpsteps = 0
    tmpresets = 0
    tstart = time.time()
    t0 = time.time()
    update_every = total_steps // 100  # 100 updates total (every 1%)
    report_every = 10  # every 10%

    buf_e = np.zeros((len(LINK_TYPES), total_steps), dtype=np.int64)
    buf_k = np.zeros((len(LINK_TYPES), total_steps, 165), dtype=np.int64)

    while steps < total_steps:
        if term or trunc:
            resets += 1
            tmpresets += 1
            obs, info = env.reset()
            term = False
            trunc = False
        else:
            act = env.random_action()
            obs, _, term, trunc, info = env.step(act)

        for lt, ldata in obs["links"].items():
            i_lt = LINK_TYPES[lt]
            buf_e[i_lt, steps] = ldata["index"].shape[1]
            buf_k[i_lt, steps] = np.bincount(ldata["index"][1], minlength=165)

        steps += 1
        tmpsteps += 1

        if steps % update_every == 0:
            percentage = (steps / total_steps) * 100
            print("\r%d%%..." % percentage, end="", flush=True)

            if steps % (update_every*report_every) == 0:
                s = time.time() - t0
                print(" steps/s: %-6.0f resets/s: %-6.2f" % (tmpsteps/s, tmpresets/s))
                tmpsteps = 0
                tmpresets = 0
                t0 = time.time()
                # avoid hiding the percent
                if percentage < 100:
                    print("\r%d%%..." % percentage, end="", flush=True)

    print("")
    print("%20s   avg   max   p99   p90   p75   p50   p25" % "Num edges (E)")
    print("-" * 65)
    for lt, ilt in LINK_TYPES.items():
        e = buf_e[ilt]
        print("%20s   %-4d  %-4d  %-4d  %-4d  %-4d  %-4d  %-4d" % (
            lt,
            np.mean(e),
            np.max(e),
            np.percentile(e, q=99),
            np.percentile(e, q=90),
            np.percentile(e, q=75),
            np.percentile(e, q=50),
            np.percentile(e, q=25),
        ))

    print("")
    print("%20s   avg   max   p99   p90   p75   p50   p25" % "Inbound edges (K)")
    print("-" * 65)
    for lt, ilt in LINK_TYPES.items():
        k = buf_k[ilt]
        kmaxs = np.max(k, axis=1)

        print("%20s   %.1f   %-4d  %-4d  %-4d  %-4d  %-4d  %-4d" % (
            lt,
            np.mean(k),
            np.max(k),
            np.percentile(kmaxs, q=99),
            np.percentile(kmaxs, q=90),
            np.percentile(kmaxs, q=75),
            np.percentile(kmaxs, q=50),
            np.percentile(kmaxs, q=25),
        ))

    # np.percentile(np.max(counters["REACH"]["k"], axis=1), q=10)
    # => 2.0
    # # i.e. 10% of the states have a max K <= 2
    # (100% of the states have a max K <= max(k))
