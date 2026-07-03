import time
import torch


class Timer:
    def __init__(self, name="default", elapsed=0, cuda_sync=True):
        self.name = name
        self.elapsed = elapsed
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
            # XXX: using time.perf_counter() is undesirable as it may be less than self.elapsed
            self._started_at = time.time() - self.elapsed
            self.elapsed = 0
        else:
            self._is_running = False
            self._started_at = 0.0
        self._time_total = 0.0

    def start(self):
        if self._is_running:
            print("WARNING: timer already started")
        self._is_running = True
        self._started_at = time.time() - self.elapsed
        self.elapsed = 0
        # print("========== START: %f (elapsed: %f)" % (self._started_at, self.elapsed))

    def stop(self):
        # print("========== STOPPING")
        self._maybe_synchronize()
        if not self._is_running:
            print("WARNING: timer already stopped")
        self._is_running = False
        self._time_total += (time.time() - self._started_at)
        # print("========== STOP: %f" % self._time_total)

    def peek(self):
        res = self._time_total
        if self._is_running:
            res += (time.time() - self._started_at)
        return max(0.0, res)
