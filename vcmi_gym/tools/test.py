import time
import logging
import sys
import os
import gymnasium as gym


def test(env_kwargs):
    env = gym.make("local/VCMI-v0")
    env.reset()
    print(env.render())

    # enchanters:
    # stay on hex 75, shoot at enemy stack 1:
    action = 2 + 75*8 + 1
    while True:
        obs, rew, term, trunc, info = env.step(action)
        # obs, rew, term, trunc, info = env.step(0)
        # logging.debug("======== obs: (hidden)")
        # logging.debug("======== rew: %s" % rew)
        # logging.debug("======== term: %s" % term)
        # logging.debug("======== trunc: %s" % trunc)
        # logging.debug("======== info: %s" % info)
        action += 1

        if env.unwrapped.last_action_n_errors == 0:
            time.sleep(1)
            print(env.render())
        if term:
            break

    print(env.unwrapped.error_summary())
