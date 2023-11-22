import time
import logging
import sys
import os
import gymnasium as gym
import itertools


def a(x, y=None, stack=0):
    if y is None:
        return x - 1
    return 3 + (y*15 + x)*8 + stack - 1


def _test(env_kwargs):
    env = gym.make("local/VCMI-v0")
    env.reset()
    print(env.render())

    # enchanters:
    # stay on hex 75, shoot at enemy stack 1:
    # action = 2 + 75*8 + 1
    actions = iter([
        a(2),
        a(0, 8, 2),
        a(3, 0),
        a(4, 0, 1),
        a(0, 8, 1),
        a(4, 0, 1),
        a(0, 8, 1),
        a(1),
        a(0, 8, 2),
        a(1),
        a(0, 8, 2),
        a(1),
        a(0, 8, 2),
        a(1),
        a(0, 8, 2),
        a(1),
        a(0, 8, 2),
        a(1),
        a(0, 8, 1),
    ])

    while True:
        action = next(actions)
        obs, rew, term, trunc, info = env.step(action)
        # obs, rew, term, trunc, info = env.step(0)
        # logging.debug("======== obs: (hidden)")
        # logging.debug("======== rew: %s" % rew)
        # logging.debug("======== term: %s" % term)
        # logging.debug("======== trunc: %s" % trunc)
        # logging.debug("======== info: %s" % info)
        # action += 1

        if env.unwrapped.last_action_was_valid:
            time.sleep(1)
            print(env.render())
        if term:
            break

    print(env.unwrapped.error_summary())


def test(env_kwargs):
    import threading
    t1 = threading.Thread(target=_test, args=(test,))
    t2 = threading.Thread(target=_test, args=(test,))
    t1.start()
    time.sleep(1)
    t2.start()
    t1.join()
    t2.join()

    # import multiprocessing
    # p1 = multiprocessing.Process(target=_test, args=(env_kwargs,))
    # p2 = multiprocessing.Process(target=_test, args=(env_kwargs,))
    # p1.start()
    # time.sleep(1)
    # p2.start()
    # p1.join()
    # p2.join()
