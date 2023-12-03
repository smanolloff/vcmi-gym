class TestHelper:
    def __init__(self, env, auto_render=True):
        self.env = env
        self.auto_render = auto_render
        print(self.env.render())

    def _maybe_render(self, action):
        retval = self.env.step(action)
        if self.auto_render:
            print(self.env.render())
        else:
            return retval

    def defend(self):
        return self._maybe_render(0)

    def wait(self):
        return self._maybe_render(1)

    # x: 1..15
    # y: 1..11
    # stack=8 means MOVE ONLY
    def move(self, x, y, stack=8):
        assert x >= 1 and x <= 15
        assert x >= 1 and y <= 11

        a = 2 + ((y-1)*15 + x-1)*8 + stack-1
        return self._maybe_render(a)
