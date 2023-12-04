import time
import gymnasium as gym


def a(x, y=None, stack=0):
    if y is None:
        return x - 1
    return 3 + (y*15 + x)*8 + stack - 1


def test(env_kwargs, actions):
    env = gym.make("local/VCMI-v0")
    env.reset()
    print(env.render())
    actions = iter(actions)

    while True:
        action = next(actions) - 1
        obs, rew, term, trunc, info = env.step(action)
        # obs, rew, term, trunc, info = env.step(0)
        # logging.debug("======== obs: (hidden)")
        # logging.debug("======== rew: %s" % rew)
        # logging.debug("======== term: %s" % term)
        # logging.debug("======== trunc: %s" % trunc)
        # logging.debug("======== info: %s" % info)
        # action += 1

        if env.unwrapped.last_action_was_valid:
            time.sleep(0.2)
            print(env.render())
        else:
            pass
            # s = "Error summary:\n"
            # for i, name in enumerate(self.errnames):
            #     s += "%25s: %d\n" % (name, self.errcounters[i])
            # print(s)

        if term:
            break

    print(env.render())
