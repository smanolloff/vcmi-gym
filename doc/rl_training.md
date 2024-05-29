# RL Training

I've spent a considerable amount of time trying to design and train an AI model
in a resource-constrained setup (personal laptop) with a (so far) promising 
results - some of my agents currently win 80% of the games vs the VCMI's Neutral bot (a.k.a. "StupidAI") and is kind of
and 60% against the "stronger" Player bot (a.k.a. "BattleAI").
It's not yet "challenging" for any decent human player :)

I've mostly been using PPO with action masking as the RL algorithm and am
experimenting with numerous neural network architectures.

I've historically used PPO implementations from
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3), but I find it
difficult to modify/customize, so I recently switched to a modified version of
[cleanrl](https://github.com/vwxyzjn/cleanrl).

I have also implemented a maskable variant of the QRDQN algorithm in hopes of
surpassing MPPO's performance, but the training results are not promising.
Other approaches such as self-attention layers, parameterized multi-head action
networks and LSTMs did not yield good results for me, but there's so much
tweaks that could be done there (which I do not have time for) that my efforts
are inconclusive.

[Population-based training](https://deepmind.google/discover/blog/population-based-training-of-neural-networks/)
is my main method of training AI agents. I also experimented with
[Population-based Bandits](https://www.anyscale.com/blog/population-based-bandits)
(a reportedly better alternative of PBT for small population sizes), but it did
not yield good results due to issues such as slow hyperparameter calculation
times, unproductive value "swings" (min->max->min->max..), etc. so I will
stick to PBT for now.

My W&B project with some of my recent training runs can be
found [here](https://wandb.ai/s-manolloff/vcmi-gym). I publish
W&B [reports](https://wandb.ai/s-manolloff/vcmi-gym/reportlist) from time to
time, but there's still quite a lot of ground to cover there.

## Loading AI models into VCMI directly

Once an AI model is trained and ready for the "real" test (where YOU play
against it 🤓), you can plug it into VCMI directly in a game where it
controls the enemy army.

For this purpose, your model must implement the `.predict(obs, mask)` and
`.get_value(obs)` methods, where `obs` and `mask` are obtained from VcmiEnv
(refer to the [Environment docs](./env_info.md)). The model must be serialized
as a [TorchScript](https://pytorch.org/docs/stable/jit_language_reference.html#language-reference)
module.
Save the trained model to a file using `torch.jit.save(torch.jit.script(agent), "agent.pt")`
then start the (modified) VCMI binary as described in the "Loading AI Models" section
[here](https://github.com/smanolloff/vcmi/blob/mmai/docs/setup_macos.md#loading-ai-models)
