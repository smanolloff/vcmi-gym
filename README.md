# VCMI Gym

`vcmi-gym` is a project which aims to create a gym-compatible reinforcement
learning environment for VCMI (the open-source recreation of
Heroes of Might & Magic III game) and use it to produce "smart" AI models for
that game.

<img src="doc/demo.gif" alt="demo">

## Project state

<p align="center"><img src="doc/Under-Construction.png" alt="UNDER CONSTRUCTION" width="300" height="250"></p>

Currently, it's only possible to train "combat" AIs (an RL environment for
adventure AIs would probably require a project on its own).

I am currently using vcmi-gym to train a VCMI battle-only AI that will
(hopefully) become "smart" enough to provide a fair combat challenge to HOMM3
players which (like me) are unhappy with the scripted AI implementations.

The project consists of two main parts:

1. A gym-compatible RL environment API (this repo) which integrates VCMI
2. A modified version of VCMI itself
  ([separate repo](https://github.com/smanolloff/vcmi)) which contains changes
  needed for RL training purposes. It is tracked as a git submodule located at
  `./vcmi_gym/envs/v0/vcmi`.

> [!NOTE]
> Being a work-in-progress, the codebase/documentation quality is
> _questionable_ as the project is evolving too frequently at this initial
> stage :)

## Getting started

`vcmi-gym` is currently tested on MacOS 14.0 and Linux - Ubuntu 22.04 LTS only.
I was unable to make it work on Windows, but any contributions in that regard
are welcome.

* For MacOS OS, please refer to [this guide](./doc/setup_macos.md)
* For Linux (Ubuntu) OS, please refer to [this guide](./doc/setup_ubuntu.md)

## Documentation

As any WIP project, the documentation is... not great.

For now there are [Environment](./doc/env_info.md) and
[RL Training](./doc/rl_training.md) docs.

_It ain't much, but it's honest work._ ;)

## Contributing

Fellow HOMM3 AI enthusiasts are more than welcome to help with this project.

Here are some basic guidelines to follow if you'd like to contribute:

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


