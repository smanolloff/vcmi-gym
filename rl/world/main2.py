import torch
import time

from .i2a import I2A
from .t10n.t10n import Reconstruction
from .p10n.p10n import Prediction
# from .t10n import t10n
# from .p10n import p10n


if __name__ == "__main__":
    from vcmi_gym.envs.v12.vcmi_env import VcmiEnv

    oldcwd = os.getcwd()
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
    i2a = I2A(
        i2a_fc_units=16,
        num_trajectories=2,
        rollout_dim=16,
        rollout_policy_fc_units=16,
        horizon=3,
        obs_processor_output_size=16,
        side=1,
        reward_step_fixed=env.reward_cfg.step_fixed,
        reward_dmg_mult=env.reward_cfg.dmg_mult,
        reward_term_mult=env.reward_cfg.term_mult,
        transition_model_file=f"{oldcwd}/hauzybxn-model.pt",
        action_prediction_model_file=f"{oldcwd}/ogyesvkb-model.pt",
        reward_prediction_model_file=f"{oldcwd}/aexhrgez-model.pt",
    )

    env.reset()
    act = env.random_action()
    obs0, rew, term, trunc, _info = env.step(act)
    done = term or trunc

    start_obs = torch.as_tensor(obs0["observation"]).unsqueeze(0)
    start_mask = torch.as_tensor(obs0["action_mask"]).unsqueeze(0)

    with torch.no_grad():
        t = time.time()
        obs = start_obs
        mask = start_mask

        for i in range(50):
            action_logits, value = i2a(obs, mask, Reconstruction.GREEDY, Prediction.GREEDY)
            probs = action_logits.masked_fill(~mask, -1e9).softmax(dim=-1)
            action = probs.multinomial(num_samples=1).squeeze(1)

            obs0, rew, term, trunc, _info = env.step(action[0].item())
            if term or trunc:
                obs0, _info = env.reset()

            obs = torch.as_tensor(obs0["observation"]).unsqueeze(0)
            mask = torch.as_tensor(obs0["action_mask"]).unsqueeze(0)
            print(".")

        elapsed = time.time() - t
        print("Elapsed: %s" % elapsed)
