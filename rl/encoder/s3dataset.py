import boto3
import torch
import numpy as np
import os
import re
import random
from torch.utils.data import IterableDataset


class S3Dataset(IterableDataset):
    def __init__(
        self,
        bucket_name,
        s3_prefix,
        cache_dir="/tmp/s3_cache",
        max_samples_in_memory=1000,
        aws_access_key=None,
        aws_secret_key=None,
        region_name="eu-north-1"
    ):
        """
        Args:
            bucket_name (str): S3 bucket name.
            cache_dir (str): Directory to cache downloaded files.
            max_samples_in_memory (int): Maximum number of samples kept in memory.
            aws_access_key (str, optional): AWS Access Key ID.
            aws_secret_key (str, optional): AWS Secret Access Key.
            region_name (str, optional): AWS region.
        """
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.cache_dir = cache_dir
        self.max_samples_in_memory = max_samples_in_memory

        print("Cache dir: %s" % self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        # AWS credentials
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region_name = region_name
        self.s3_client = None  # Will be initialized per worker

        # Track completed epochs
        self.epoch_count = 0

        self.s3_client = None
        self._build_filelist()

    def _init_s3_client(self):
        """ Lazily initializes the S3 client in each worker process """
        return boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region_name
        )

    def _build_filelist(self):
        """ Lists available .npz files in S3 matching the given pattern """

        # XXX: dont set self.s3_client here (forking fails later)
        s3_client = self._init_s3_client()

        types = ["obs", "mask", "done", "action", "reward"]
        prefix_counters = {}
        regex = re.compile(fr"{self.s3_prefix}/({'|'.join(types)})-(.*)\.npz")
        request = {"Bucket": self.bucket_name, "Prefix": self.s3_prefix}

        while True:
            response = s3_client.list_objects_v2(**request)
            all_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.npz')]

            for f in sorted(all_files):
                m = regex.match(f)
                assert m, f"Unexpected filename in S3: {f}"
                prefix = m[2]

                if prefix not in prefix_counters:
                    prefix_counters[prefix] = 1
                else:
                    prefix_counters[prefix] += 1

            if "NextContinuationToken" in response:
                request["ContinuationToken"] = response["NextContinuationToken"]
            else:
                break

        prefixes = [prefix for prefix, counter in prefix_counters.items() if counter == len(types)]

        dropped = [prefix for prefix in prefix_counters.keys() if prefix not in prefixes]
        if dropped:
            print("WARNING: dropped %d incomplete prefixes: %s" % (len(dropped), dropped))

        # DEBUG
        print("XXXXXXX: USING JUST 3 PREFIXES (DEBUG)")
        prefixes = prefixes[:3]
        # /DEBUG

        print("Found %d sample packs" % len(prefixes))

        self.types = types
        self.prefixes = prefixes

    def _get_worker_prefixes(self, worker_id, num_workers):
        """ Assigns distinct file chunks to each worker """
        return self.prefixes[worker_id::num_workers]

    def _download_file(self, file_key):
        """ Downloads an .npz file from S3 to the cache directory if not already present """
        if not self.s3_client:
            self.s3_client = self._init_s3_client()

        local_path = os.path.join(self.cache_dir, os.path.basename(file_key))
        if os.path.exists(local_path):
            print("Using cached file %s" % local_path)
        else:
            print("Downloading %s" % file_key)
            self.s3_client.download_file(self.bucket_name, file_key, local_path)
        return local_path

    def _stream_samples(self, worker_id, worker_prefixes):
        """ Reads and yields samples from S3 .npz files while managing memory constraints """
        samples = {}

        while True:
            random.shuffle(worker_prefixes)  # Randomize on each full pass
            for prefix in worker_prefixes:
                for t in self.types:
                    s3_path = f"{self.s3_prefix}/{t}-{prefix}.npz"
                    local_path = self._download_file(s3_path)
                    samples[t] = np.load(local_path)['arr_0']

                lengths = [len(s) for s in samples.values()]
                assert all(lengths[0] == l for l in lengths[1:]), lengths

                for sample_group in zip(samples["obs"], samples["mask"], samples["done"], samples["action"], samples["reward"]):
                    yield sample_group

            self.epoch_count += 1  # Track how many times we’ve exhausted S3 files
            print(f"Epoch {self.epoch_count} completed. Restarting dataset.")

    def __iter__(self):
        """ Creates an iterable dataset that streams from S3 """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-process
            worker_prefixes = self.prefixes
        else:  # Multi-worker: Assign distinct files
            worker_prefixes = self._get_worker_prefixes(worker_info.id, worker_info.num_workers)

        return self._stream_samples(worker_info.id, worker_prefixes)


if __name__ == "__main__":
    dataset = S3Dataset(
        bucket_name="vcmi-gym",
        s3_prefix="v8",
        cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache")),
        aws_access_key=os.environ["AWS_ACCESS_KEY"],
        aws_secret_key=os.environ["AWS_SECRET_KEY"],
        region_name="eu-north-1"
    )

    # XXX: prefetch_factor means=N:
    #      always keep N*num_workers*batch_size records preloaded
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, num_workers=1, prefetch_factor=10)
    for x in dataloader:
        import ipdb; ipdb.set_trace()  # noqa
        print("yee")
