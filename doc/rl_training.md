# RL Training

I've spent a considerable amount of time trying to design and train an AI model
in a resource-constrained setup (personal laptop) with moderate success - it
currently wins against the VCMI's Neutral AI (a.k.a. "StupidAI") and is kind of
50/50 against VCMI's "stronger" Player AI (a.k.a. "BattleAI").
It's by no means "challenging" for any decent human player, so there's much
to be desired :)

I've mostly been using PPO with action masking as the RL algorithm and am
experimenting with numerous neural network architectures. So far I have
achieved best results with a simple 1-layer 2D convolutional layer, followed
by a FC-1024 layer with batch normalization an LeakyReLU activations and it's
basically showing a 60+% winrate vs VCMI's Neutral AI (StupidAI) and
is struggling at around 50% vs. VCMI's Player AI (BattleAI):

<p>
<img src="rl-stupidai.png" width="400">
<img src="rl-battleai.png" width="400">
</p>

I've historically used PPO implementations from
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3), but I find it
difficult to modify/customize, so I recently switched to a modified version of
[cleanrl](https://github.com/vwxyzjn/cleanrl).

I also attempted to implement masking for SB3-contrib's QRDQN implementation,
but that algorithm is too heavy and learning was too slow/non-existent (might
also be a problem with my modification), so I abandoned that.
Self-attention layers, parameterized multi-head action networks and LSTMs did
not yield good results for me, but there's so much tweaks that could be done
there (which I do not have time for) that my efforts there were mostly
inconclusive.

I also applied a
[Population-based training](https://deepmind.google/discover/blog/population-based-training-of-neural-networks/)
approach at the beginning (using
[ray-project](https://github.com/ray-project/ray)'s PBT implementation) and it
did provide some good results, however I have currently abandoned due to issues
with the run controller (on some occations it kills the environments without
waiting for proper cleanup which is *essential* in this case). I ended up
coding my own quick-and-dirty "watchdog" script which takes off a lot of the
burden related to dealing with ray's services for simple cases like mine.
There's a ton of other things I tried, but I don't feel like sharing all of my
failures here :) If this project gains enough traction, someone will eventually
find the right formula. 

My W&B project with some of my recent training runs can be
found [here](https://wandb.ai/s-manolloff/vcmi-gym).
Older training runs will not appear there, as they were in another (quite
messy) W&B project.

## Loading AI models into VCMI directly

Once an AI model is trained and ready for the "real" test (where YOU play
against it 🤓), you can plug it into VCMI directly in a game where it
controls the enemy army.

For this purpose, your model must implement a single `.predict(obs, mask)`
method where `obs` and `mask` are obtained from VcmiEnv (refer to
the [Environment docs](./env_info.md)). Save your
trained model to a file using `torch.save(...)` starting the
(modified) VCMI binary as described in the "Loading AI Models" section
[here](https://github.com/smanolloff/vcmi/blob/mmai/docs/setup_macos.md#loading-ai-models)
