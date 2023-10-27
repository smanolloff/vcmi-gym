from build import connector
import time
import asyncio
import threading

# attempts to fix issue in connector-threading.py
# by using ThreadPoolExecutor instead of vanilla thread


async def start_vcmi(event, shared_obj):
    print("[PY] (start_vcmi) called with: event=%s | shared_obj=%s" % (event, shared_obj))
    shared_obj.append("start_vcmi")

    def pycallback(vcmi_retval):
        print("[PY] (pycallback) called with: vcmi_retval=%d" % vcmi_retval)
        shared_obj.append("pycallback")

        print("[PY] (pycallback) sleep(1)")
        time.sleep(1)

        print("[PY] (pycallback) event.set()")
        event.set()
        print("[PY] (pycallback) return")

    # print("[PY] (start_vcmi) call connector.start_vcmi(...)")
    # connector.start_vcmi(pycallback, 1)

    # print("[PY] (start_vcmi) threading.Thread(target=connector.start_vcmi, args=...)")
    # t = threading.Thread(target=connector.start_vcmi, args=(pycallback, 1))
    # t.run()

    print("[PY] (start_vcmi) loop.run_in_executor(...)")
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, connector.start_vcmi, pycallback, 1)

    print("[PY] (start_vcmi) event.wait()")
    await event.wait()

    print("[PY] (start_vcmi) sleep(1)")
    time.sleep(1)

    print("[PY] (start_vcmi) return shared_obj")
    return shared_obj


def main():
    shared_obj = ["main"]
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
