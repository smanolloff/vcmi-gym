# vcmi-gym setup guide (MacOS)

> [!IMPORTANT]
> VCMI requires data files from the original "Heroes 3: Shadow of Death" or
> "Heroes 3: Complete" editions. <br>Make sure you have access to those files
> before you proceed.

The setup guide below is tested with Python 3.10.12 on MacOS 14.0.

### Checkout code

```bash
$ git clone --recurse-submodules https://github.com/smanolloff/vcmi-gym.git
$ export VCMI_GYM_DIR="$PWD/vcmi-gym"
$ export VCMI_DIR="$VCMI_GYM_DIR/vcmi_gym/envs/v0/vcmi"
```

### Python env and deps

To avoid polluting your system with vcmi-gym dependencies, it's best to create
a [Python virtual env](https://docs.python.org/3/library/venv.html):

```bash
$ cd "$VCMI_GYM_DIR"
$ python3 -m venv .venv
```

The newly created virtual environment is contained within the local `.venv`
directory. Instruct your terminal to use it for this session:

```bash
$ source .venv/bin/activate
```
> [!NOTE]
> A `(.venv)` appears in your prompt to inform you that the python virtual
> environment is active. In this tutorial, all commands that require an active
> python virtual env will be indicated with a `(.venv) $` prompt.

```bash
(.venv)$ pip install -r requirements.txt
```

### Build VCMI

Please follow the instructions in [this guide](https://github.com/smanolloff/vcmi/blob/mmai/docs/setup_macos.md).

### Build vcmi-gym C++ libs

These libraries are the "link" between the gym env and VCMI itself.

```bash
$ cd "$VCMI_GYM_DIR/envs/v0/connector"
$ conan install . \
    --install-folder=conan-generated \
    --no-imports \
    --build=missing \
    --profile:build=default \
    --profile:host=default

$ cmake --fresh -S . -B build -Wno-dev \
    -D CMAKE_TOOLCHAIN_FILE=conan-generated/conan_toolchain.cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0

$ cmake --build build/
```

### Gym maps

Auto-generated maps for the purposes of training combat AIs must be symlinked
in order to make them visible in VCMI:

```bash
(.venv)$ ln -s "$VCMI_GYM_DIR/maps/gym" "$HOME/Library/Application Support/vcmi/Maps/gym"
```

### Manual test

Open up the Python REPL:

```bash
(.venv)$ python

Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

```python
>>> import vcmi_gym  # may take some time (5-10s)
>>> env = vcmi_gym.VcmiEnv("ai/P1.vmap")
>>> print(env.render())
>>> obs, rew, term, trunc, info = env.step(0)   # perform a "WAIT" action
>>> print(env.render())
```

For manual testing, inconvenient to interact with the env directly. The `TestHelper` is a handy class for the purpose:

```bash
>>> th = vcmi_gym.TestHelper(env)
>>> th.wait()           # perform a "wait" action
>>> th.move(1, 1)       # move to (x,y)
>>> th.melee(5, 1, 3)   # move to (x,y) and attack at a direction 0..11 (see hexaction.h in VCMI)
```

> [!NOTE]
> Running VCMI as a gym environment makes it **impossible** to use the VCMI GUI.
> This hard limitation comes from the SDL renderer (used by VCMI for the GUI)
> which can only work in the main process thread (in this case, the Python
> interpreter itself is the main thread and VCMI is launched as a child thread).

