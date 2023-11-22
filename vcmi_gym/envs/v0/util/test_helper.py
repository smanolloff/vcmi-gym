class TestHelper:
    def __init__(self, env):
        self.env = env
        print(self.env.render())

    def defend(self):
        self.env.step(0)
        print(self.env.render())

    def wait(self):
        self.env.step(1)
        print(self.env.render())

    # x: 1..15
    # y: 1..11
    def move(self, x, y, stack=0):
        assert x >= 1 and x <= 15
        assert x >= 1 and y <= 11

        a = 2 + ((y-1)*15 + x-1)*8 + stack
        self.env.step(a)
        print(self.env.render())
