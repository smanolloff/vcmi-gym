import torch

from .world import I2A
# from .t10n import t10n
# from .p10n import p10n


if __name__ == "__main__":
    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv

    env = VcmiEnv(
        # mapname="gym/generated/evaluation/8x512.vmap",
        mapname="gym/A1.vmap",
        opponent="BattleAI",
        swap_sides=0,
        role="defender",
        # random_heroes=1,
        # random_obstacles=1,
        # town_chance=20,
        # warmachine_chance=30,
        # random_stack_chance=65,
        # random_terrain_chance=100,
        # tight_formation_chance=30,
    )

    assert env.role == "defender"
    i2a = I2A(2, 3)

    env.reset()
    act = env.random_action()
    obs0, rew, term, trunc, _info = env.step(act)
    done = term or trunc

    start_obs = torch.as_tensor(obs0["observation"]).unsqueeze(0)
    start_mask = torch.as_tensor(obs0["action_mask"]).unsqueeze(0)

    with torch.no_grad():
        res = i2a(start_obs, start_mask)
        import ipdb; ipdb.set_trace()  # noqa
        print("")
