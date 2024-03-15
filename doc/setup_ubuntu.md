# vcmi-gym setup guide (Ubuntu)

> [!IMPORTANT]
> VCMI requires data files from the original "Heroes 3: Shadow of Death" or
> "Heroes 3: Complete" editions. <br>Make sure you have access to those files
> before you proceed.

The setup guide below is tested with Python 3.10.12 on Ubuntu 22.04.

### Checkout code

Please refer to the [Checkout code](./setup_macos.md#checkout-code)
instructions for MacOS.

### Build VCMI

Please follow the instructions in [this guide](https://github.com/smanolloff/vcmi/blob/mmai/docs/setup_macos.md).

### Build vcmi-gym C++ libs

Custom-made libraries that "connect" VCMI with the gym env:

```bash
$ sudo apt install pybind11-dev
$ cd "$VCMI_GYM_DIR/envs/v0/connector"
$ cmake --fresh -S . -B build -Wno-dev \
    -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=1

$ cmake --build build/
```
### Python env and deps

Please refer to the [Python env and deps](./setup_macos.md#python-env-and-deps)
instructions for MacOS.

### Vcmi-gym maps

Auto-generated maps for the purposes of training combat AIs must be symlinked
in order to make them visible in VCMI:

```bash
$ ln -s "$VCMI_GYM_DIR/maps/gym" "${XDG_DATA_HOME:-$HOME/.local/share}/vcmi/Maps/gym"
```

### Manual test

Please refer to the [Manual test](./setup_macos.md#manual-test)
instructions for MacOS.
