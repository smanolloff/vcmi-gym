from build import connector
import time
import asyncio
import threading


# uses threading, does not work:

# $ python connector.py
# [PY] (main) asyncio start
# [PY] (start_vcmi) called with: event=<asyncio.locks.Event object at 0x105511390 [unset]> | shared_obj=['main']
# [PY] (start_vcmi) threading.Thread(target=connector.start_vcmi, args=...)
# ^^^^ thread blocks here (CPP is actually blocking)
# [CPP] (start_vcmi) called with pycallback=?, value=1
# [CPP] (start_vcmi) acquire lock
# [CPP] (start_vcmi) boost::thread(lambda)
# [CPP] (start_vcmi) sleep 1s
# [CPP] (lambda) called with value=1
# [CPP] (lambda) acquire lock
# [CPP] (start_vcmi) cond.wait()
# [CPP] (lambda) sleep 1s
# [CPP] (lambda) call pycallback(value+1)
# ^^^^ deadlock: CPP waits py and py waits CPP

async def start_vcmi(event, shared_obj):
    print("[PY] (start_vcmi) called with: event=%s | shared_obj=%s" % (event, shared_obj))
    shared_obj[1] = 1

    def pycallback(vcmi_retval):
        print("[PY] (pycallback) called with: vcmi_retval=%d" % vcmi_retval)
        shared_obj[2] = 2

        print("[PY] (pycallback) sleep(1)")
        time.sleep(1)

        print("[PY] (pycallback) event.set()")
        event.set()
        print("[PY] (pycallback) return")

    print("[PY] (start_vcmi) call connector.start_vcmi(...)")
    connector.start_vcmi(pycallback, 1)

    # print("[PY] (start_vcmi) threading.Thread(target=connector.start_vcmi, args=...)")
    # t = threading.Thread(target=connector.start_vcmi, args=(pycallback, 1))
    # t.run()

    print("[PY] (start_vcmi) event.wait()")
    await event.wait()

    print("[PY] (start_vcmi) sleep(1)")
    time.sleep(1)

    print("[PY] (start_vcmi) return shared_obj")
    return shared_obj


def main():
    shared_obj = [0, 0, 0, 0, 0];
    event = asyncio.Event()

    print("[PY] (main) asyncio start")
    retval = asyncio.run(start_vcmi(event, shared_obj))

    print("[PY] (main) asyncio done")
    print("[PY] (main) retval=%s" % retval)
    print("[PY] (main) shared_obj=%s" % shared_obj)
    print("[PY] (main) retval is shared_obj: %s" % (retval is shared_obj))


if __name__ == '__main__':
    main()
    print("[PY] (__main__) sleep 1")
    time.sleep(1)
    main()
    # cppconn = ctypes.CDLL('build/libconnector.dylib')
    # cppconn.cppconn(python_callback)
