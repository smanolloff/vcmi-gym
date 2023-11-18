import multiprocessing
import time
import ctypes

PyState = ctypes.c_float * 3


class PyResult(ctypes.Structure):
    _fields_ = [
        ("baba", ctypes.c_char * 5)
    ]


class Simo():
    def __init__(self):
        print("init: start")
        self.x = "x"
        self.z = multiprocessing.Value(ctypes.c_int, 3)
        p = multiprocessing.Process(target=self.meth, args="y")
        p.start()
        self.x = "xXXXXX"
        print("init: sleep(2)")
        time.sleep(2)
        print("init: sleep(2) - woke up")
        print("init: x is: %s" % self.x)

        # Y is uninitialized!
        # print("init: y is: %s" % self.y)

        print("init: z is: %s" % self.z.value)
        print("init: end")

    def meth(self, _y):
        print("meth: start")
        print("meth: sleep(1)")
        time.sleep(1)
        print("meth: sleep(1) - woke up")
        print("meth: x is: %s" % self.x)
        self.y = _y
        print("meth: y is: %s" % self.y)
        self.z.value = 3333333
        print("meth: z is: %s" % self.z.value)
        print("meth: end")


if __name__ == "__main__":
    # s = Simo()
    # s.meth("YYYY")
    # print(s.y)

    v_result = multiprocessing.Value(PyResult)
    v_result.state = PyState(1,2,3)
    v_result.baba = "adadsadsdas"
    for (i, x) in enumerate(v_result.state):
        print("%d: %s" % (i, x))
    print(v_result.baba)
