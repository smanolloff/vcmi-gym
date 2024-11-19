# XXXXXXXXXX:
# tf.compat.v1.enable_eager_execution() can NOT be set
# when running the script with "python -m path.to.module"
# It MUST be started with "python path/to/module.py"
# ...

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import rl.ray.mdreamerv3.pbt_debug_main

if __name__ == "__main__":
    rl.ray.mdreamerv3.pbt_debug_main.main()
