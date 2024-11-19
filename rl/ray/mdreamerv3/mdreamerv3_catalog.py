from ray.rllib.algorithms.dreamerv3.dreamerv3_catalog import DreamerV3Catalog
from ray.rllib.algorithms.dreamerv3.tf.models.components.vector_decoder import VectorDecoder
from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.utils import override

from .mdreamerv3_encoder import MDreamerV3_Encoder


class MDreamerV3_Catalog(DreamerV3Catalog):
    def __init__(self, observation_space, action_space, model_config_dict):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

        self.model_size = self._model_config_dict["model_size"]
        self.is_img_space = False
        self.is_gray_scale = False

    @override(DreamerV3Catalog)
    def build_encoder(self, framework: str) -> Encoder:
        return MDreamerV3_Encoder()

    def build_decoder(self, framework: str) -> Model:
        return VectorDecoder(
            model_size=self.model_size,
            observation_space=self.observation_space,
        )
