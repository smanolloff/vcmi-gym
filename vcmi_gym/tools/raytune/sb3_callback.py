import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
import wandb

from ... import InfoDict


class SB3Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_episodes = 0
        self.rollouts = 0
        self.ep_rew_mean = 0

        # wdb tables are logged only once, at end of the iteration
        # (ie. once every config["rollouts_per_iteration"])
        self.wdb_tables = {}
        self.uncommitted_log = False

    def _on_step(self):
        self.rollout_episodes += self.locals["dones"].sum()

    def _on_rollout_start(self):
        if self.uncommitted_log:
            # Must log *something* to commit the rest
            # Since timesteps are preserved between checkpoint loads,
            # mark this as a trial-related metric
            wandb.log({"trial/num_timesteps": self.num_timesteps})
            self.uncommitted_log = False

    def _on_rollout_end(self):
        self.rollouts += 1
        wdb_log = {"rollout/n_episodes": self.rollout_episodes}
        self.rollout_episodes = 0

        if self.rollouts % self.locals["log_interval"] != 0:
            return

        for k in InfoDict.SCALAR_VALUES:
            v = safe_mean([ep_info[k] for ep_info in self.model.ep_info_buffer])
            self.model.logger.record(f"{k}", v)
            wdb_log[k] = v

        # From here on it's W&B stuff only

        wdb_log["rollout/count"] = self.rollouts

        # Also add sb3's Monitor info keys: "r" (reward) and "l" (length)
        # (these are already recorded to TB by sb3, but not in W&B)
        self.ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_rew_mean"] = self.ep_rew_mean

        v = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        wdb_log["rollout/ep_len_mean"] = v

        for (k, columns) in InfoDict.D1_ARRAY_VALUES.items():
            action_types_vec_2d = [ep_info[k] for ep_info in self.model.ep_info_buffer]
            ary = np.mean(action_types_vec_2d, axis=0)

            # In SB3's logger, Tensor objects are logged as a Histogram
            # https://github.com/DLR-RM/stable-baselines3/blob/v1.8.0/stable_baselines3/common/logger.py#L412
            # NOT logging this to TB, it's not visualized well there
            # tb_data = torch.as_tensor(ary)
            # self.model.logger.record(f"user/{k}", tb_data)

            # In W&B, we need to unpivot into a name/count table
            # NOTE: reserved column names: "id", "name", "_step" and "color"
            wk = f"table/{k}"
            rotated = [list(row) for row in zip(columns, ary)]
            if wk not in self.wdb_tables:
                self.wdb_tables[wk] = wandb.Table(columns=["key", "value"])

            wb_table = self.wdb_tables[wk]
            for row in rotated:
                wb_table.add_data(*row)

        for k in InfoDict.D2_ARRAY_VALUES:
            action_types_vec_3d = [ep_info[k] for ep_info in self.model.ep_info_buffer]
            ary_2d = np.mean(action_types_vec_3d, axis=0)

            wk = f"table/{k}"
            if wk not in self.wdb_tables:
                # Also log the "rollout" so that inter-process logs (which are different _step)
                # can be aggregated if needed
                self.wdb_tables[wk] = wandb.Table(columns=["x", "y", "value"])

            wb_table = self.wdb_tables[wk]

            for (y, row) in enumerate(ary_2d):
                for (x, cell) in enumerate(row):
                    wb_table.add_data(x, y, cell)

        # Commit will be done either:
        #   a) on_rollout_start()
        #   b) by the trainer (if this is the last rollout)
        wandb.log(wdb_log, commit=False)
        self.uncommitted_log = True
