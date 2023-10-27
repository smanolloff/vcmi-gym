from build import connector
import time
import asyncio
import threading
import numpy as np


async def act_vcmi(cppcb, action):
    print("[PY] (act_vcmi) called")
    event = asyncio.Event()
    state_wrapper = {"state": "TODO_1", "cppcb": "TODO_1"}

    # cb to be called from VCMI client thread
    def pycallback_1(state, cppcb):
        print("[PY] (pycallback_1) called with: state=%d, cppcb=%s" % (state, cppcb))
        print("[PY] (pycallback_1) state.getA(): %s, state.getB(): %s" % state.getA(), state.getB())

        print("[PY] (pycallback_1) sleep(1)")
        time.sleep(1)

        # TODO:
        # must somehow pass this state to the other thread
        # however, simply assigning it to a shared variable would
        # cause segfaults? (unconfirmed)
        # state_wrapper["state"] = state
        # state_wrapper["cppcb"] = cppcb

        # state and cppcb are now available => unblock act_vcmi
        print("[PY] (pycallback_1) event.set()")
        event.set()

        print("[PY] (pycallback_1) return")

    cppaction = connector.Action()
    cppaction.setA(str(action))
    cppaction.setB(action)

    print("[PY] (act_vcmi) call cppcb(%s, %s)" % (pycallback_1, cppaction))
    cppcb(pycallback_1, cppaction)

    # wait until cppcb calls pycallback_1, setting state and cppcb
    print("[PY] (act_vcmi) event.wait()")
    await event.wait()

    print("[PY] (act_vcmi) state: %s" % state)

    return state["state"], state["cppcb"]


async def start_vcmi(event):
    print("[PY] (start_vcmi) called")

    event = asyncio.Event()
    state_wrapper = {"state": "TODO", "cppcb": "TODO"}

    # cb to be called from VCMI client thread
    def pycb(state, cppcb):
        print("[PY] (pycb) called with: state=%d, cppcb=%s" % (state, cppcb))
        print("[PY] (pycb) state.getA(): %s, state.getB(): %s" % state.getA(), state.getB())

        print("[PY] (pycb) sleep(1)")
        time.sleep(1)

        # TODO:
        # must somehow pass this state to the other thread
        # however, simply assigning it to a shared variable would
        # cause segfaults? (unconfirmed)
        # state_wrapper["state"] = state
        # state_wrapper["cppcb"] = cppcb

        # state and cppcb are now available => unblock start_vcmi
        print("[PY] (pycb) event.set()")
        event.set()

        print("[PY] (pycb) return")

    # print("[PY] (start_vcmi) call connector.start_vcmi(...)")
    # connector.start_vcmi(pycb)

    print("[PY] (start_vcmi) threading.Thread(target=connector.start_vcmi, args=(pycb))")
    t = threading.Thread(target=connector.start_vcmi, args=(pycb))
    t.run()

    # wait until connector.start_vcmi calls pycb, setting state and cppcb
    print("[PY] (start_vcmi) event.wait()")
    await event.wait()

    print("[PY] (start_vcmi) state: %s" % state)

    return state["state"], state["cppcb"]


def main():
    shared_obj = [0, 0, 0, 0, 0];

    #
    # Simulates env = VcmiEnv()
    #
    print("[PY] (main) asyncio start")
    state, cppcb = asyncio.run(start_vcmi())
    print("[PY] (main) asyncio done")
    print("[PY] (main) state=%s, cppcb=%s" % (state, cppcb))

    #
    # Simulates 3 steps
    #
    for i in range(3):
        action = np.array([i, i+1, i+2], dtype=np.float32)
        state, cppcb = asyncio.run(act_vcmi(cppcb, action))

    # return from .step()


if __name__ == '__main__':
    main()
    print("[PY] (__main__) sleep 1")
    time.sleep(1)
    main()
    # cppconn = ctypes.CDLL('build/libconnector.dylib')
    # cppconn.cppconn(python_callback)
