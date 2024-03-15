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

<a href="rl-stupidai.png">
<a href="rl-battleai.png">

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

There are also the limitations of the environment itself (e.g. having to
restart the entire game in order to switch army compositions) are negatively
impacting the training performance. Hopefully they will one day be dealt
with -- the full [list of features](https://github.com/smanolloff/vcmi) I'd
like to see implemented in VCMI shows there's still much work to be done
and your help would be greatly appreciated in this regard :)

My messy and unannotated W&B project with some of my older training runs can be
found [here](https://wandb.ai/s-manolloff/vcmi). Recent training runs will not
appear there, as I have created a separate W&B project (not public yet), with
the hope to keep it more organized and suitable and then share it.
