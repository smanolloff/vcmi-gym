import gymnasium as gym
import re
from .. import TestHelper

def user_action(th):
    hlp = """
Actions:
    w           wait
    d           defend
    x y         move to x,y (x=1..15, y=1..11)
    x y N       move to x,y and attack stack #N
    h           show this help
"""

    while True:
        words = input("Choose an action ['h'=help]: ").split()

        if len(words) == 0:
            pass
        elif words[0] == "h":
            print(hlp)
            continue
        elif words[0] == "w" and len(words) == 1:
            return th.wait()
        elif words[0] == "d" and len(words) == 1:
            return th.defend()
        elif len(words) == 2 \
                and re.match(r"^([1-9]|(1[0-5]))$", words[0]) \
                and re.match(r"^([1-9]|(1[01]))$", words[1]):
            return th.move(int(words[0]), int(words[1]))
        elif len(words) == 3 \
                and re.match(r"^([1-9]|(1[0-5]))$", words[0]) \
                and re.match(r"^([1-9]|(1[01]))$", words[1]) \
                and re.match(r"^[1-7]$", words[2]):
            return th.move(int(words[0]), int(words[1]), int(words[2]))

        print("Invalid action.")
        print(hlp)


def play():
    env = gym.make("local/VCMI-v0")
    env.reset()
    th = TestHelper(env, auto_render=False)

    while True:
        obs, rew, term, trunc, info = user_action(th)
        print(env.render())
        if term:
            break
