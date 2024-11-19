from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np


def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    initializable_layers = (
        tf.keras.layers.Dense,
    )

    if isinstance(layer, initializable_layers):
        orthogonal_initializer = tf.keras.initializers.Orthogonal(gain=gain)
        constant_initializer = tf.keras.initializers.Constant(value=bias_const)

        if layer.kernel is not None:
            layer.kernel.assign(orthogonal_initializer(shape=layer.kernel.shape))
        if layer.bias is not None:
            layer.bias.assign(constant_initializer(shape=layer.bias.shape))

    for sublayer in getattr(layer, "layers", []):
        layer_init(sublayer, gain, bias_const)

    return layer


def build_layer(spec):
    kwargs = dict(spec)  # copy
    t = kwargs.pop("t")
    layer_cls = getattr(tf.keras.layers, t, None) or globals()[t]
    return layer_cls(**kwargs)


@dataclass
class NetConfig:
    attention: dict = None
    features_extractor1_misc: list[dict] = field(default_factory=list)
    features_extractor1_stacks: list[dict] = field(default_factory=list)
    features_extractor1_hexes: list[dict] = field(default_factory=list)
    features_extractor2: list[dict] = field(default_factory=list)
    actor: dict = field(default_factory=dict)
    critic: dict = field(default_factory=dict)


class Split(tf.keras.layers.Layer):
    def __init__(self, split_size, axis):
        super().__init__()
        self.split_size = split_size
        self.axis = axis

    def call(self, x):
        return tf.split(x, num_or_size_splits=self.split_size, axis=self.axis)

    # def __repr__(self):
    #     return f"{self.__class__.__name__}(dim={self.dim}, split_size={self.split_size})"


class MDreamerV3_Encoder(tf.keras.Model):
    def __init__(self, action_space, observation_space, obs_dims, netconfig):
        super().__init__(name="encoder")

        assert isinstance(obs_dims, dict)
        assert list(obs_dims.keys()) == ["misc", "stacks", "hexes"]  # order is important

        self.obs_splitter = Split(list(obs_dims.values()), axis=1)

        self.features_extractor1_misc = tf.keras.Sequential(name="features_extractor1_misc")
        for spec in netconfig.features_extractor1_misc:
            layer = build_layer(spec)
            self.features_extractor1_misc.add(layer)

        # dummy input to initialize layers
        dummy_outputs = [self.features_extractor1_misc(tf.random.normal([1, obs_dims["misc"]]))]

        for layer in self.features_extractor1_misc.layers:
            layer_init(layer)

        self.features_extractor1_stacks = tf.keras.Sequential(name="features_extractor1_stacks")
        self.features_extractor1_stacks.add(tf.keras.layers.Reshape([20, obs_dims["stacks"] // 20]))

        for spec in netconfig.features_extractor1_stacks:
            layer = build_layer(spec)
            self.features_extractor1_stacks.add(layer)

        # dummy input to initialize layers
        dummy_outputs.append(self.features_extractor1_stacks(tf.random.normal([1, obs_dims["stacks"]])))

        self.features_extractor1_stacks.build(input_shape=[None, obs_dims["stacks"]])
        for layer in self.features_extractor1_stacks.layers:
            layer_init(layer)

        self.features_extractor1_hexes = tf.keras.Sequential(name="features_extractor1_hexes")
        self.features_extractor1_hexes.add(tf.keras.layers.Reshape([165, obs_dims["hexes"] // 165]))

        for spec in netconfig.features_extractor1_hexes:
            layer = build_layer(spec)
            self.features_extractor1_hexes.add(layer)

        # dummy input to initialize layers
        dummy_outputs.append(self.features_extractor1_hexes(tf.random.normal([1, obs_dims["hexes"]])))

        for layer in self.features_extractor1_hexes.layers:
            layer_init(layer)

        self.features_extractor2 = tf.keras.Sequential(name="features_extractor2")
        for spec in netconfig.features_extractor2:
            layer = build_layer(spec)
            self.features_extractor2.add(layer)

        # dummy input to initialize layers
        self.features_extractor2(tf.concat(dummy_outputs, axis=1))

        for layer in self.features_extractor2.layers:
            layer_init(layer)

    def call(self, x):
        misc, stacks, hexes = self.obs_splitter(x)
        fmisc = self.features_extractor1_misc(misc)
        fstacks = self.features_extractor1_stacks(stacks)
        fhexes = self.features_extractor1_hexes(hexes)
        features1 = tf.concat([fmisc, fstacks, fhexes], axis=1)
        return self.features_extractor2(features1)
