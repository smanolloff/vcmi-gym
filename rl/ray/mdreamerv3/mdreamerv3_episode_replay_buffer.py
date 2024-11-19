from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.utils.spaces.space_utils import batch


class MDreamerV3_EpisodeReplayBuffer(EpisodeReplayBuffer):
    # XXX: fix replay buffer sampling with dict obs space
    def _sample_batch(self, *args, **kwargs):
        res = super()._sample_batch(*args, **kwargs)
        res["obs"] = batch([batch(list(o)) for o in res["obs"]])
        return res
