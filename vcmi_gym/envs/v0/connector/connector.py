from build import connector
import time


def python_callback(v):
    print(">>> in connector.py with %d" % v)


if __name__ == '__main__':
    # cppconn = ctypes.CDLL('build/libconnector.dylib')
    # cppconn.cppconn(python_callback)
    connector.cpp_function(python_callback, 1)
    print("cpp function returned, sleep now...")

    time.sleep(10)
