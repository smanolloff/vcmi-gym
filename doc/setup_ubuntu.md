# vcmi-gym setup guide (Ubuntu)

The setup guide below is tested on Ubuntu 22.04 fresh install.

1. Checkout vcmi-gym (with vcmi as a submodule)

    ```bash
    $ git clone --recurse-submodules https://github.com/smanolloff/vcmi-gym.git
    $ export VCMI_GYM_DIR="$PWD/vcmi-gym"
    $ export VCMI_DIR="$VCMI_GYM_DIR/vcmi_gym/envs/v0/vcmi"
    ```

1. Build VCMI as per [these instructions](TODO:link to VCMI instructions)

1. Link ai maps:
    ```bash
    $ ln -s "$VCMI_GYM_DIR/testing/maps" "${XDG_DATA_HOME:-$HOME/.local/share}/vcmi/Maps/ai"
    ```

1. Build C++ libs: connector, loader:

    ```bash
    $ cd "$VCMI_GYM_DIR/envs/v0/connector"
    $ sudo apt-get install pybind11-dev
    $ cmake --fresh -S . -B build -Wno-dev \
        -D CMAKE_BUILD_TYPE=Debug \
        -D CMAKE_EXPORT_COMPILE_COMMANDS=1

    $ rm lib/*.dylib
    $ ln -s ../../vcmi/build/bin/libmyclient.so lib/
    $ cmake --build build/
    $ ln -st . build/connexport*.so
    ```

1. Setup a [Python virtual env](https://docs.python.org/3/library/venv.html) (tested with python 3.10.12):

    ```bash
    $ cd "$VCMI_GYM_DIR"
    $ sudo apt-get install python3.10-venv
    $ python3 -m venv .venv
    ```

    The newly created virtual environment is contained within the local `.venv` directory.
    Instruct your terminal to use it for this session:

    ```bash
    $ source .venv/bin/activate
    ```

    A `(.venv)` will appear next to your terminal prompt to indicate an active virtual env.

    ```bash
    (.venv)$ sed -i 's/^tensorflow-macos/#&/' requirements.lock
    (.venv)$ pip install -r requirements.lock
    ```

5. Test installation

    ```bash
    $ cd "$VCMI_GYM_DIR"
    $ source .venv/bin/activate
    ```

    ```bash
    (.venv)$ python

    Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.

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
