import torch


class BufferBase:
    def __init__(self, logger, dataloader, dim_obs, n_actions, name="buffer", device=torch.device("cpu")):
        self.logger = logger
        self.dataloader = dataloader
        # num_workers can be set to 0 for debugging
        self.capacity = dataloader.batch_size * max(dataloader.num_workers, 1)
        self.name = name
        self.device = device

        self.containers = {
            "obs": torch.zeros((self.capacity, dim_obs), dtype=torch.float32, device=device),
            "mask": torch.zeros((self.capacity, n_actions), dtype=torch.float32, device=device),
            "reward": torch.zeros((self.capacity,), dtype=torch.float32, device=device),
            "done": torch.zeros((self.capacity,), dtype=torch.float32, device=device),
            "action": torch.zeros((self.capacity,), dtype=torch.int64, device=device)
        }

        self.index = 0
        self.full = False

        # XXX: dirty hack to prevent (obs, obs_next) from different workers
        #   1. assumes dataloader fetches `batch_size` samples from 1 worker
        #       (instead of e.g. round robin worker for each sample)
        #   2. assumes buffer.capacity % dataloader.batch_size == 0
        self.worker_cutoffs = [i * dataloader.batch_size - 1 for i in range(1, dataloader.num_workers)]

    def load_samples(self, dataloader):
        self.logger.debug("Loading observations...")

        # This is technically not needed, but is easier to benchmark
        # when the batch sizes for adding and iterating are the same
        assert self.index == 0, f"{self.index} == 0"

        self.full = False
        while not self.full:
            self.add_batch(next(dataloader))

        assert self.index == 0, f"{self.index} == 0"
        self.logger.debug(f"Loaded {self.capacity} observations")

    def add(self, data):
        self.containers["obs"][self.index] = torch.as_tensor(data.obs, dtype=torch.float32, device=self.device)
        self.containers["mask"][self.index] = torch.as_tensor(data.mask, dtype=torch.float32, device=self.device)
        self.containers["reward"][self.index] = torch.as_tensor(data.reward, dtype=torch.float32, device=self.device)
        self.containers["done"][self.index] = torch.as_tensor(data.done, dtype=torch.float32, device=self.device)
        self.containers["action"][self.index] = torch.as_tensor(data.action, dtype=torch.int64, device=self.device)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def add_batch(self, data):
        batch_size = data.obs.shape[0]
        start = self.index
        end = self.index + batch_size

        assert end <= self.capacity, f"{end} <= {self.capacity}"
        assert self.index % batch_size == 0, f"{self.index} % {batch_size} == 0"
        assert self.capacity % batch_size == 0, f"{self.capacity} % {batch_size} == 0"

        self.containers["obs"][start:end] = torch.as_tensor(data.obs, dtype=torch.float32, device=self.device)
        self.containers["mask"][start:end] = torch.as_tensor(data.mask, dtype=torch.float32, device=self.device)
        self.containers["reward"][start:end] = torch.as_tensor(data.reward, dtype=torch.float32, device=self.device)
        self.containers["done"][start:end] = torch.as_tensor(data.done, dtype=torch.float32, device=self.device)
        self.containers["action"][start:end] = torch.as_tensor(data.action, dtype=torch.int64, device=self.device)

        self.index = end
        if self.index == self.capacity:
            self.index = 0
            self.full = True

    def sample(self, batch_size):
        raise NotImplementedError()

    def sample_iter(self, batch_size):
        raise NotImplementedError()
