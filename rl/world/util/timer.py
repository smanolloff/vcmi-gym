import time
import torch


class Timer:
    def __init__(self, cuda_sync=True):
        self._cuda = torch.cuda.is_available() and cuda_sync
        self.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _maybe_synchronize(self):
        if self._cuda:
            # wait for cuda async operations to complete
            torch.cuda.synchronize()

    def reset(self, start=False):
        self._maybe_synchronize()
        if start:
            self._is_running = True
            self._started_at = time.perf_counter()
        else:
            self._is_running = False
            self._started_at = 0.0
        self._time_total = 0.0

    def start(self):
        if self._is_running:
            print("WARNING: timer already started")
        self._is_running = True
        self._started_at = time.perf_counter()
        # print("========== START: %f" % self._started_at)

    def stop(self):
        # print("========== STOPPING")
        self._maybe_synchronize()
        if not self._is_running:
            print("WARNING: timer already stopped")
        self._is_running = False
        self._time_total += (time.perf_counter() - self._started_at)
        # print("========== STOP: %f" % self._time_total)

    def peek(self):
        res = self._time_total
        if self._is_running:
            res += (time.perf_counter() - self._started_at)
        return res
