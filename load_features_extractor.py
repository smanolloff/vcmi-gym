from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
import vcmi_gym
import torch

learner_kwargs = {
    'batch_size': 64, 'n_epochs': 10, 'gamma': 0.8, 'gae_lambda': 0.8,
    'clip_range': 0.4, 'normalize_advantage': True, 'ent_coef': 0.001,
    'vf_coef': 0.6, 'max_grad_norm': 1.5, 'learning_rate': 0.0006,
    'n_steps': 170, 'policy': 'MlpPolicy', 'policy_kwargs': {
        'net_arch': [64, 64],
        'activation_fn': torch.nn.ReLU,
        'features_extractor_class': vcmi_gym.VcmiFeaturesExtractor,
        'features_extractor_kwargs': {
            'output_dim': 1024,
            'activation': 'ReLU',
            'layers': [
                {'t': 'Conv2d', 'out_channels': 32, 'kernel_size': [1, 15], 'stride': [1, 15], 'padding': 0}
            ]
        },
        'optimizer_class': torch.optim.AdamW,
        'optimizer_kwargs': {'eps': 1e-05, 'weight_decay': 0}
    }
}

venv = make_vec_env("VCMI-v0", env_kwargs={"mapname": "ai/generated/B431.vmap"})
model = MaskablePPO(env=venv, **learner_kwargs)

loadfile = "data/MPPO-32q24d2n/32q24d2n_1705008488/model-1705048846.zip"
_data, params, _pytorch_variables = load_from_zip_file(loadfile)
prefix = "features_extractor."
features_extractor_params = dict(
    (k.removeprefix(prefix), v) for (k, v) in params["policy"].items() if k.startswith(prefix)
)

model.policy.features_extractor.load_state_dict(features_extractor_params, strict=True)
