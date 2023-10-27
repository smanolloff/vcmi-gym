from build import connector
import time
import asyncio
import multiprocessing

# WIP version
# attempts to fix issue in connector-threading.py
# by using multiprocessing instead of threading
# Hoever, problems arise:
# asyncio's Event is *different* than multiprocessing's Event
# > the multiprocessing.Event() (ie. a PyCapsule object) can't be
#   passed to the child process (is not C-serializable)
#


async def start_vcmi(event_mp, event_io, values):
    print("[PY] (start_vcmi) called with: values=%s" % (event, values[:]))
    values[1] = 1

    def pycallback(cb_values):
        print("[PY] (pycallback) called with: cb_values=%s" % cb_values[:])
        print("[PY] (pycallback) values=%s" % values[:])
        print("[PY] (pycallback) values is cb_values: %s" % (values is cb_values))

        print("[PY] (pycallback) set cb_values[4]=666")
        cb_values[4] = 2

        print("[PY] (pycallback) sleep(1)")
        time.sleep(1)

        print("[PY] (pycallback) event.set()")
        event.set()
        print("[PY] (pycallback) return")

    # print("[PY] (start_vcmi) call connector.start_vcmi(...)")
    # connector.start_vcmi(pycallback, 1)

    print("[PY] (start_vcmi) Process(target=connector.start_vcmi, args=...)")
    p = multiprocessing.Process(target=connector.start_vcmi, args=(pycallback, values))
    p.start()

    print("[PY] (start_vcmi) event.wait()")
    await event.wait()

    print("[PY] (start_vcmi) sleep(1)")
    time.sleep(1)

    print("[PY] (start_vcmi) return values")
    return values


def main():
    # types: https://docs.python.org/3/library/ctypes.html
    # 5-element array of ints:
    values = multiprocessing.Array(ctypes.c_int, 5)
    event_mp = multiprocessing.Event()
    event_io = asyncio.Event()

    print("[PY] (main) asyncio start")
    retval = asyncio.run(start_vcmi(event_mp, event_io, values))

    print("[PY] (main) asyncio done")
    print("[PY] (main) retval=%s" % retval)
    print("[PY] (main) values=%s" % values)
    print("[PY] (main) retval is values: %s" % (retval is values))


if __name__ == '__main__':
    main()
    print("[PY] (__main__) sleep 1")
    time.sleep(1)
    main()
    # cppconn = ctypes.CDLL('build/libconnector.dylib')
    # cppconn.cppconn(python_callback)
