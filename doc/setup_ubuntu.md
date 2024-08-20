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
$ cd "$VCMIGYM/vcmi_gym/connectors"
$ sudo apt install libboost-all-dev
$ cmake -S . -B rel -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=0

$ cmake --build rel/
```
### Python env and deps

Please refer to the [Python env and deps](./setup_macos.md#python-env-and-deps)
instructions for MacOS.

### Vcmi-gym maps

Auto-generated maps for the purposes of training combat AIs must be symlinked
in order to make them visible in VCMI:

```bash
$ ln -s "$VCMIGYM/maps/gym" "${XDG_DATA_HOME:-$HOME/.local/share}/vcmi/Maps/gym"
```

### Manual test

Please refer to the [Manual test](./setup_macos.md#manual-test)
instructions for MacOS.
