from ..common import common_main, util
from . import MIMPALA_Algorithm, MIMPALA_Config


# Silence here in order to supress messages from local runners
# Silence once more in algo init to supress messages from remote runners
util.silence_log_noise()

if __name__ == "__main__":
    common_main.main(MIMPALA_Config, MIMPALA_Algorithm, __package__)
