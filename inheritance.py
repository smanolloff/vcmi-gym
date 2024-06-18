

class A:
    class Kur:
        def __init__(self):
            print("init: KUR")

    TEST = "(A.TEST)"

    def __init__(self):
        self.__class__.test()
        print(self.__class__.TEST)
        self.__class__.Kur()
        self.__class__.test2()

    @staticmethod
    def test():
        print("in A.test")

    @classmethod
    def test2(cls):
        print("in A.test2. cls=%s", cls.__name__)


class B(A):
    class Kur:
        def __init__(self):
            print("init: KUR2")

    TEST = "(B.TEST)"

    @staticmethod
    def test():
        print("in B.test")


a = A()
b = B()
