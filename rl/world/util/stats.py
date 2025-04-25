import numpy as np
import torch


class Stats:
    def __init__(self, model, device):
        self.device = device

        # Simple counter. 1 sample = 1 obs
        # i.e. the counter for each hex feature will be 165*num_samples
        self.num_samples = 0

        # Number of updates, should correspond to training iteration
        self.iteration = 0


        # Not implemented
        # # Store [mean, var] for each continuous feature
        # # Shape: (N_CONT_FEATURES, 2)
        # self.continuous = {
        #     "global": torch.zeros(*model.rel_index_global["continuous"].shape, 2, device=device),
        #     "player": torch.zeros(*model.rel_index_player["continuous"].shape, 2, device=device),
        #     "hex": torch.zeros(*model.rel_index_hex["continuous"].shape, 2, device=device),
        # }

        # self.cont_nullbit = {
        #     "global": torch.zeros(*model.rel_index_global["cont_nullbit"].shape, 2, dtype=torch.int64, device=device),
        #     "player": torch.zeros(*model.rel_index_player["cont_nullbit"].shape, 2, dtype=torch.int64, device=device),
        #     "hex": torch.zeros(*model.rel_index_hex["cont_nullbit"].shape, 2, dtype=torch.int64, device=device),
        # }

        # # Store [n_ones_class0, n_ones_class1, ...]  for each categorical feature
        # # Python list with N_CAT_FEATURES elements
        # # Each element has shape: (N_CLASSES, 2), where N_CLASSES varies
        # # e.g.
        # # [
        # #  [n_ones_F1_class0, n_ones_F1_class1, n_ones_F1_class2],
        # #  [n_ones_F2_class0, n_ones_F2_class2, n_ones_F2_class3, n_ones_F2_class4],
        # #  ...etc
        # # ]
        # self.categoricals = {
        #     "global": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_global["categoricals"]],
        #     "player": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_player["categoricals"]],
        #     "hex": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_hex["categoricals"]],
        # }

        # self.binaries = {
        #     "global": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_global["binaries"]],
        #     "player": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_player["binaries"]],
        #     "hex": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_hex["binaries"]],
        # }

        # self.thresholds = {
        #     "global": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_global["thresholds"]],
        #     "player": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_player["thresholds"]],
        #     "hex": [torch.zeros(ind.shape, dtype=torch.int64, device=device) for ind in model.rel_index_hex["thresholds"]],
        # }

    def export_data(self):
        return {
            "iteration": self.iteration,
            # Not implemented
            # "num_samples": self.num_samples,
            # "continuous": self.continuous,
            # "cont_nullbit": self.cont_nullbit,
            # "binaries": self.binaries,
            # "categoricals": self.categoricals
        }

    def load_state_dict(self, data):
        pass
        self.iteration = data["iteration"]
        self.num_samples = data["num_samples"]
        # Not implemented
        # self.continuous = data["continuous"]
        # self.cont_nullbit = data["cont_nullbit"]
        # self.categoricals = data["categoricals"]
        # self.binaries = data["binaries"]
        # self.thresholds = data["thresholds"]

    def update(self, buffer, model):
        with torch.no_grad():
            self._update(buffer, model)

    def _update(self, buffer, model):
        self.num_samples += buffer.capacity

        # Not implemented: cont_nullbit, binaries, thresholds

        # self.continuous["global"][:, 0] = model.encoder_global_continuous[0].running_mean
        # self.continuous["global"][:, 1] = model.encoder_global_continuous[0].running_var
        # self.continuous["player"][:, 0] = model.encoder_player_continuous[1].running_mean
        # self.continuous["player"][:, 1] = model.encoder_player_continuous[1].running_var
        # self.continuous["hex"][:, 0] = model.encoder_hex_continuous[1].running_mean
        # self.continuous["hex"][:, 1] = model.encoder_hex_continuous[1].running_var

        # obs = buffer.obs_buffer

        # # stat.add_(obs[:, ind].sum(0).round().long())
        # values_global = obs[:, model.abs_index["global"]["binary"]].round().long()
        # self.binary["global"][:, 0] += values_global.sum(0)
        # self.binary["global"][:, 1] += np.prod(values_global.shape)

        # values_player = obs[:, model.abs_index["player"]["binary"]].flatten(end_dim=1).round().long()
        # self.binary["player"][:, 0] += values_player.sum(0)
        # self.binary["player"][:, 1] += np.prod(values_player.shape)

        # values_hex = obs[:, model.abs_index["hex"]["binary"]].flatten(end_dim=1).round().long()
        # self.binary["hex"][:, 0] += values_hex.sum(0)
        # self.binary["hex"][:, 1] += np.prod(values_hex.shape)

        # for ind, stat in zip(model.abs_index["global"]["categoricals"], self.categoricals["global"]):
        #     stat.add_(obs[:, ind].round().long().sum(0))

        # for ind, stat in zip(model.abs_index["player"]["categoricals"], self.categoricals["player"]):
        #     stat.add_(obs[:, ind].flatten(end_dim=1).round().long().sum(0))

        # for ind, stat in zip(model.abs_index["hex"]["categoricals"], self.categoricals["hex"]):
        #     stat.add_(obs[:, ind].flatten(end_dim=1).round().long().sum(0))

    def compute_loss_weights(self):
        pass
        # Not implemented: cont_nullbit, binaries, thresholds
        # weights = {
        #     "binary": {
        #         "global": torch.tensor(0., device=self.device),
        #         "player": torch.tensor(0., device=self.device),
        #         "hex": torch.tensor(0., device=self.device)
        #     },
        #     "categoricals": {
        #         "global": [],
        #         "player": [],
        #         "hex": []
        #     },
        # }

        # # NOTE: Clamping weights to prevent huge weights for binaries
        # # which are never positive (e.g. SLEEPING stack flag)
        # for type in weights["binary"].keys():
        #     s = self.binary[type]
        #     if len(s) == 0:
        #         continue
        #     num_positives = s[:, 0]
        #     num_negatives = s[:, 1] - num_positives
        #     pos_weights = num_negatives / num_positives
        #     weights["binary"][type] = pos_weights.clamp(max=100)

        # # NOTE: Computing weights only for labels that have appeared
        # # to prevent huge weights for labels which never occur
        # # (e.g. hex.IS_REAR) from making the other weights very small
        # for type, cat_weights in weights["categoricals"].items():
        #     for cat_stats in self.categoricals[type]:
        #         w = torch.zeros(cat_stats.shape, dtype=torch.float32, device=self.device)
        #         mask = cat_stats > 0
        #         masked_stats = cat_stats[mask].float()
        #         w[mask] = masked_stats.mean() / masked_stats
        #         cat_weights.append(w.clamp(max=100))

        # return weights
