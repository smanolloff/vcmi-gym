import botocore.config
import boto3
import torch
import time
import queue
import os
import re
import random
import glob
import logging
import numpy as np
from torch.utils.data import IterableDataset

from .timer import Timer
from .dataset_vcmi import Data, noop_functor


class DatasetS3(IterableDataset):
    def __init__(
        self,
        logger,
        bucket_name,
        s3_dir,
        cache_dir,
        cached_files_max=None,
        aws_access_key=None,
        aws_secret_key=None,
        region_name="eu-north-1",
        shuffle=False,
        split_ratio=1.0,
        split_side=0,
        metric_queue=None,
        metric_report_interval=5,
        mw_functor=noop_functor
    ):
        """
        Args:
            bucket_name (str): S3 bucket name.
            cache_dir (str): Directory to cache downloaded files.
            aws_access_key (str, optional): AWS Access Key ID.
            aws_secret_key (str, optional): AWS Secret Access Key.
            region_name (str, optional): AWS region.
        """
        self.bucket_name = bucket_name
        self.s3_dir = s3_dir
        self.cache_dir = cache_dir
        self.cached_files_max = cached_files_max
        self.shuffle = shuffle
        self.logger = logger
        self.split_ratio = split_ratio
        self.split_side = split_side
        self.metric_queue = metric_queue
        self.metric_report_interval = metric_report_interval
        self.metric_reported_at = time.time()

        self.timer_all = Timer()
        self.timer_idle = Timer()
        self.mw_functor = mw_functor

        assert split_ratio >= 0 and split_ratio <= 1, split_ratio
        assert split_side in [0, 1], split_side

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
            region_name=self.region_name,
            config=botocore.config.Config(connect_timeout=10, read_timeout=30)
        )

    def _build_filelist(self):
        """ Lists available .npz files in S3 matching the given pattern """

        # XXX: dont set self.s3_client here (forking fails later)
        s3_client = self._init_s3_client()
        regex = re.compile(fr"{self.s3_dir}/transitions-(.*)\.npz")
        request = {"Bucket": self.bucket_name, "Prefix": f"{self.s3_dir}/"}
        all_keys = []

        while True:
            response = s3_client.list_objects_v2(**request)
            all_keys.extend([obj['Key'] for obj in response.get('Contents', []) if regex.match(obj['Key'])])

            if "NextContinuationToken" in response:
                request["ContinuationToken"] = response["NextContinuationToken"]
            else:
                break

        if os.getenv("S3_DEBUG"):
            self.logger.info("XXXXXXX: USING JUST 3 PREFIXES (DEBUG)")
            all_keys = all_keys[:3]

        self.logger.info("Found %d sample packs" % len(all_keys))

        if self.split_ratio < 1:
            self.logger.info("will split using ratio %.2f" % self.split_ratio)
            split_idx = int(len(all_keys) * self.split_ratio)
            self.all_keys = all_keys[:split_idx] if self.split_side == 0 else all_keys[split_idx:]
            self.logger.info("Sample packs after split: %d" % len(self.all_keys))
        else:
            assert self.split_side == 0, "split side is %d, but split_ratio is %f" % (self.split_side, self.split_ratio)
            self.all_keys = all_keys

    def _get_worker_keys(self, worker_id, num_workers):
        return self.all_keys[worker_id::num_workers]

    def _download_file(self, file_key):
        """ Downloads an .npz file from S3 to the cache directory if not already present """
        if not self.s3_client:
            self.s3_client = self._init_s3_client()

        local_path = os.path.join(self.cache_dir, file_key)
        if os.path.exists(local_path):
            self.logger.debug("Using cached file %s" % local_path)
        else:
            self.logger.warn("Downloading %s ..." % file_key)
            self.s3_client.download_file(self.bucket_name, file_key, local_path)
            self.logger.debug("Download complete: %s" % file_key)

            if self.cached_files_max is not None:
                # Keep the most recent 100 files and delete the rest
                files = sorted(glob.glob(os.path.join(self.cache_dir, "*")), key=os.path.getmtime, reverse=True)
                for file in files[self.cached_files_max:]:
                    try:
                        os.remove(file)
                        self.logger.info(f"Deleting: {file}")
                    except Exception as e:
                        self.logger.warn(f"Error deleting {file}: {e}")

        return local_path

    def _stream_samples(self, worker_keys):
        middleware = self.mw_functor()

        with self.timer_all:
            while True:
                if self.shuffle:
                    random.shuffle(worker_keys)

                for s3_key in worker_keys:
                    samples = dict(np.load(self._download_file(s3_key)))

                    for (o, m, r, d, a) in zip(samples["obs"], samples["mask"], samples["reward"], samples["done"], samples["action"]):
                        with self.timer_idle:
                            data = middleware(Data(obs=o, mask=m, reward=r, done=d, action=a))

                        if data is not None:
                            with self.timer_idle:
                                yield data

                        if self.metric_queue and time.time() - self.metric_reported_at > self.metric_report_interval:
                            self.metric_reported_at = time.time()
                            try:
                                self.metric_queue.put(self.utilization(), block=False)
                            except queue.Full:
                                logger.warn("Failed to report metric (queue full)")

                self.epoch_count += 1  # Track how many times we’ve exhausted S3 files
                self.logger.info(f"Epoch {self.epoch_count} completed. Restarting dataset.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-process
            worker_keys = self.all_keys
        else:  # Multi-worker: Assign distinct files
            worker_keys = self._get_worker_keys(worker_info.id, worker_info.num_workers)

        return self._stream_samples(worker_keys)

    def utilization(self):
        if self.timer_all.peek() == 0:
            return 0
        return 1 - (self.timer_idle.peek() / self.timer_all.peek())


if __name__ == "__main__":
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    dataset = DatasetS3(
        logger=logger,
        bucket_name="vcmi-gym",
        s3_dir="v11/4x1024",
        shuffle=False,
        cache_dir=os.path.abspath("data/.s3_cache"),
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
