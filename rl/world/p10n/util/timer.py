import time

class Timer:
    def __init__(self):
        self.reset()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def reset(self):
        self._is_running = False
        self._time_total = 0.0
        self._started_at = 0.0

    def start(self):
        if self._is_running:
            print("WARNING: timer already started")
        self._is_running = True
        self._started_at = time.perf_counter()
        # print("========== START: %f" % self._started_at)

    def stop(self):
        # print("========== STOPPING")
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
