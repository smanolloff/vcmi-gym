import numpy as np
import time
import gymnasium as gym
import threading
import enum

from .util import log

DEBUG = True

def tracelog(func):
    if not DEBUG:
        return func

    def wrapper(*args, **kwargs):
        this = args[0]
        this.debug("Start: %s (args=%s, kwargs=%s)" % (func.__name__, args[1:], kwargs))
        result = func(*args, **kwargs)
        this.debug("End: %s (return %s)" % (func.__name__, result))
        return result

    return wrapper


class State(Enum):
    RESET = enum.auto()
    REG_A = enum.auto()
    REG_B = enum.auto()
    OBS_A = enum.auto()


class Side():
    # Must correspond to VCMI's MMAI::Export::Side enum
    ATTACKER = 0
    DEFENDER = 1


class DualController():
    def __init__(self, env):
        self.env = env
        self.client_a = None
        self.client_b = None
        self.state = State.RESET
        self.cond = threading.Condition()
        self.logger = log.get_logger("DualController", "debug")
        self.sides = {Side.ATTACKER: None, Side.DEFENDER: None}

    def debug(self, msg):
        self.logger.debug("[%s][%s] %s" % (threading.current_thread().ident, self.state.name, msg))

    def transition(self, newstate):
        self.debug("%s => %s" % (self.state.name, newstate.name))
        self.state = newstate

    @tracelog
    def reset(self):
        self.cond.acquire()

        match self.state:
            case State.RESET:
                assert self.client_a is None, "expected None, got: %s" % self.client_a
                self.transition(State.REG_A)
                self.cond.wait()

                assert self.state == State.REG_B, "expected REG_B, got: %s" % self.state.name
                obs, info = self.env.reset()
                self.client_a = Side(info["side"])
                self.transition(State.OBS_A)
                return self.client_a, obs

            case State.REG_A:
                assert self.client_b is None, "expected None, got: %s" % self.client_b
                self.transition(State.REG_B)
                self.cond.wait()

            case _:
                raise Exception("Cannot reset while in state: %s" % self.state.name)

    @tracelog
    def step(self, side, action):
        self.cond.acquire()

        match side:
            case Side.ATTACKER:
                assert self.state == State.OBS_A, "expected OBS_A, got: %s" % self.state.name
                obs, rew, term, trunc, info = self.env.reset()
                oside = Side(info["side"])

                self.transition(State.OBS_)
                self.cond.wait()

                self.client_a = Side(info["side"])
                self.transition(State.OBS_A)
                return self.client_a, obs

            case State.REG_A:
                self.transition(State.REG_B)
                self.cond.wait()


class DualEnvClient(gym.Env):
    def __init__(self, controller):
        self.controller = controller

    def reset(self, *args, **kwargs):
        self.side, obs = self.controller.reset()

        return self.env.reset(*args, **kwargs)


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.enabled:
            return obs, reward, terminated, truncated, info

        self.n_steps += 1
        self.total_reward += reward

        flags = self.env.action_cmdflags[action]

        if terminated:
            keys = "XXXX"
        else:
            keys = ""
            keys += "Q" if flags & WSProto.CMD_K_Q else "."
            keys += "W" if flags & WSProto.CMD_K_W else "."
            keys += "O" if flags & WSProto.CMD_K_O else "."
            keys += "P" if flags & WSProto.CMD_K_P else "."

        ds = info["distance"] - self.last_distance
        dt = info["time"] - self.last_time
        v = ds / dt

        print(
            "%-05d | %-4s | %-4s | %-6sm | %-6s m/s | %-6s | %6s.%s s | %-6s"
            % (
                self.n_steps,
                action,
                keys,
                round(info["distance"], 1),
                round(v, 1),
                round(reward, 2),
                int(info["time"]),
                str(info["time"] - int(info["time"]))[2:3],
                round(self.total_reward, 2),
            )
        )

        elapsed_time = time.time() - self.start_time

        if terminated:
            print("Game over")
            print("Elapsed time (real): %.1f seconds" % elapsed_time)
            print("Elapsed time (game): %.1f seconds" % info["time"])
            print("Distance ran: %.1f m" % info["distance"])
            print("Average speed: %.1f m/s" % info["avgspeed"])
            print("FPS: %.1f" % (self.n_steps / elapsed_time))

        self.last_distance = info["distance"]
        self.last_time = info["time"]

        return obs, reward, terminated, truncated, info

    # Convenience methods for controlling verbosity
    def disable_verbose_wrapper(self):
        self.enabled = False

    def enable_verbose_wrapper(self):
        self.enabled = True
