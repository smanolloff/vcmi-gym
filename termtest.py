import atexit
import signal
# import multiprocessing
import time
import sys


def atexit1():
    print("*** atexit (1) ***")


def handle_exit(signum, frame):
    print("*** received %s ***" % signum)
    sys.exit(0)


if __name__ == '__main__':
    atexit.register(atexit1)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    time.sleep(100)
