from sb3_contrib import MaskablePPO
import connexport


# XXX: maybe import VcmiEnv and load offset from there?
ACTION_OFFSET = 1
OBS_SHAPE = (1, 11, 15 * connexport.get_n_hex_attrs())


class Loader:
    class MPPO:
        def __init__(self, file):
            self.model = MaskablePPO.load(file)
            # self.obs = np.ndarray((2310,), dtype=np.float32)
            # self.actmasks = np.ndarray((1652,), dtype=np.bool)

        def predict(self, obs, actmasks):
            # np.copyto(self.obs, obs)
            # np.copyto(self.actmasks, actmasks)

            action, _states = self.model.predict(
                obs.reshape(OBS_SHAPE),
                action_masks=actmasks[ACTION_OFFSET:]
            )

            return action + ACTION_OFFSET
