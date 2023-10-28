import logging
# import asyncio
import threading
import numpy as np
import time

from build import connector

# async def act_vcmi(cppcb, action):
#     logging.info("called")
#     event = asyncio.Event()
#     state_wrapper = {"state": "TODO_1", "cppcb": "TODO_1"}

#     # cb to be called from VCMI client thread
#     def pycallback_1(state, cppcb):
#         logging.info("called with: state=%d, cppcb=%s" % (state, cppcb))
#         logging.info("state.getA(): %s, state.getB(): %s" % state.getA(), state.getB())

#         logging.info("sleep(1)")
#         time.sleep(1)

#         # TODO:
#         # must somehow pass this state to the other thread
#         # however, simply assigning it to a shared variable would
#         # cause segfaults? (unconfirmed)
#         # state_wrapper["state"] = state
#         # state_wrapper["cppcb"] = cppcb

#         # state and cppcb are now available => unblock act_vcmi
#         logging.info("event.set()")
#         event.set()

#         logging.info("return")

#     cppaction = connector.Action()
#     cppaction.setA(str(action))
#     cppaction.setB(action)

#     logging.info("call cppcb(%s, %s)" % (pycallback_1, cppaction))
#     cppcb(pycallback_1, cppaction)

#     # wait until cppcb calls pycallback_1, setting state and cppcb
#     logging.info("event.wait()")
#     await event.wait()

#     logging.info("state: %s" % state)

#     return state["state"], state["cppcb"]


def start_vcmi(event, state_wrapper, pycb, pycbinit):
    logging.info("start")

    # np1 = np.array([1,2,3.0])
    # np2 = np.array([1,2,5.0])
    # t = threading.Thread(target=connector.add_arrays, args=(np1, np2), daemon=True)
    # t.run()

    logging.info("Create thread for connector.start_vcmi()")
    t = threading.Thread(target=connector.start_vcmi, args=(pycb, pycbinit), daemon=True)
    t.start()

    # wait until connector.start_vcmi calls pycb, setting state and cppcb
    logging.info("event.wait()")
    event.wait()

    logging.info("return")


def start():
    state_wrapper = {"state": "TODO", "cppcb": "TODO"}
    event = threading.Event()

    # cb to be called from VCMI client thread
    def pycb(state):
        logging.info("start: state=%s" % state)
        logging.info("state.getA(): %s, state.getB(): %s" % (state.getA(), state.getB()))

        logging.info("sleep(1)")
        time.sleep(1)

        # XXX: Danger: SIGSERV?
        logging.info("Set shared var: state_wrapper['state']")
        state_wrapper["state"] = state

        logging.info("event.set()")
        event.set()

        logging.info("return")

    # cb to be called from VCMI client thread on init only
    def pycbinit(cppcb):
        logging.info("start: cppcb=%s" % cppcb)

        # XXX: Danger: SIGSERV?
        logging.info("Set shared vars: state_wrapper['cppcb']")
        state_wrapper["cppcb"] = cppcb

        logging.info("return")

    logging.info("start")
    start_vcmi(event, state_wrapper, pycb, pycbinit)
    logging.info("state=%s, cppcb=%s" % (state_wrapper["state"], state_wrapper["cppcb"]))

    logging.info("return")
    return state_wrapper["state"], state_wrapper["cppcb"]
