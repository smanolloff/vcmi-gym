from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
config = (
    DreamerV3Config()
    .environment("CartPole-v1")
    .training(
        model_size="XS",
        training_ratio=1,
        # TODO
        model={
            "batch_size_B": 1,
            "batch_length_T": 1,
            "horizon_H": 1,
            "gamma": 0.997,
            "model_size": "XS",
        },
    )
)

config = config.learners(num_learners=0)
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()
# algo.train()
del algo
