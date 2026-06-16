import time
from vcmi_gym.envs.v15.vcmi_env import VcmiEnv


if __name__ == "__main__":
    steps = 0
    counter = 0

    def play(env, steps):
        global counter
        while counter < steps:
            obs, rew, term, trunc, info = env.step(env.random_action())
            if term or trunc:
                env.reset()
            if counter % 1000 == 0:
                print("%d/%d (%d%%) ..." % (counter, steps, counter // 1000))
            counter += 1

    env = VcmiEnv(
        "gym/generated/4096/4x1024.vmap",
        role="defender",
        max_steps=500,
        vcmi_loglevel_global="error",
        vcmi_loglevel_ai="error",
        vcmienv_loglevel="WARN",
        random_heroes=1,
        random_obstacles=1,
        random_terrain_chance=100,
        tight_formation_chance=0,
        town_chance=10,
        random_stack_chance=20,  # makes armies unbalanced
        warmachine_chance=40,
        mana_min=0,
        mana_max=0,
        random_primary_skills=0,
        reward_step_fixed=-0.5,

        # These require BATTLE_ROUND in obs
        reward_prog_base=0.1,
        reward_prog_trigger=9,
        reward_prog_exponent=2,
        reward_prog_limit=15,

        reward_dmg_mult=0.01,
        reward_term_mult=0.01,
        reward_relval_mult=0.01,
        swap_sides=0,
        # With DualVecEnv, all timeouts must be the same (large enough)
        user_timeout=2400,
        vcmi_timeout=2400,
        boot_timeout=2400,
    )

    ts = time.time()
    play(env, 100_000)
    elapsed = time.time() - ts
    print("Elapsed: %.3fs, steps: %d, steps/s: %.2f" % (elapsed, counter, counter/elapsed))
