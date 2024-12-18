# VCMI Gym

`vcmi-gym` is a project which aims to create a gym-compatible reinforcement
learning environment for VCMI (the open-source recreation of
Heroes of Might & Magic III game) along with implementations of RL algorithms
and other supplementary code (orchestration, hyperparameter tuning, observability)
needed to produce VCMI combat AI models.

<img src="doc/demo.gif" alt="demo">

## Project state

Models trained by vcmi-gym can be loaded in VCMI through the changes proposed
in this pull request: https://github.com/vcmi/vcmi/pull/4788. When the VCMI
team accepts the contribution, vcmi-gym's models will become readily available
in VCMI through the VCMI mod ecosystem. 

Training new models is an ongoing effort and does not block players from using
the already trained MMAI models once released, as they enhance gameplay
experience by adding engaging and unpredictable behaviour of the enemy troops
during battle.

## Project architecture

A high-level overview of the `vcmi-gym` project is given in the below diagram:

<img src="doc/components.png" alt="components" height="500px">

The W&B external component is optional. VCMI (the game itself) is required --
it is a fork of VCMI with some modifications for RL training purposes and is
managed in a [separate repo](https://github.com/smanolloff/vcmi), tracked as a
[git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) located
at `./vcmi`.

## Getting started

### Installation

A step-by-step setup guide can be found below:
* [MacOS](./doc/setup_macos.md)
* [Linux](./doc/setup_ubuntu.md)
* No setup guide for Windows :(. Contributions in this regard are welcome.

### Environment documentation

Please refer to [this document](./doc/env_info.md) for more information about
the RL environment.

### RL training setup

Please refer to [this document](./doc/rl_training.md) for information about 
the RL training setup and tools used in this project.

### Connector docs

Please refer to [this document](./doc/rl_training.md) for information about 
the Connector component.

## Contributing

Fellow HOMM3 AI enthusiasts are more than welcome to help with this project.
There is a lot of headroom for improvement, be it in the form of NN
architectures, RL algorithm implementations, hyperparameter search, reward
shaping, etc. The preferred approach is to submit a Pull request, but if you
have stumbled upon a bug which you can't fix yourself, submitting an issue can
help me (and others) fix it. You can also help with the ongoing AI training
process by "plugging in" your own piece of hardware into the mix. 

### Submitting an issue

Please check for existing issues and verify that your issue is not already
submitted. If it is, it's highly recommended to add to that issue with your
reports.

When submitting a new issue, please be as detailed as possible - OS and Python
versions, what did you do, what did you expect to happen, and what actually
happened.

### Submitting a Pull Request

1. Find an existing issue to work on or follow "Submitting an issue" to first
  create one that you're also going to fix.
  Make sure to notify that you're working on a fix for the issue you picked.
1. Branch out from latest `main` and organize your code changes there.
1. Commit, push to your branch and submit a PR to `main`.

### Contributing with RL training

If you have spare hardware and would like to help with this project, please
<a href="mailto:smanolloff@gmail.com">reach out</a> to me - together, we can
ponder on putting it to good use: e.g. for training new models (GPU-bound
task), evaluating existing models (CPU-bound task), or creating and rebalancing
new training maps (CPU and HDD-bound task).
