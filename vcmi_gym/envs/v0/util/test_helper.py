from .pyconnector import (
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
    N_HEX_ATTRS,
)


class TestHelper:
    def __init__(self, env, auto_render=True):
        self.env = env
        self.auto_render = auto_render
        self.obs, _ = env.reset()
        print(self.env.render())

    def render(self):
        print(self.env.render())

    def _maybe_render(self, action):
        retval = self.env.step(action)
        self.obs = retval[0]
        if self.auto_render:
            print(self.env.render())
        else:
            return retval

    def wait(self):
        return self._maybe_render(0)

    def _move_action(self, x, y):
        assert x >= 1 and x <= 15
        assert x >= 1 and y <= 11
        return N_NONHEX_ACTIONS + ((y-1)*15 + x-1)*N_HEX_ACTIONS - 1

    def _shoot_action(self, x, y):
        return self._move_action(x, y) + 1  # SHOOT is 1 after MOVE

    def _melee_action(self, x, y, direction):
        return self._shoot_action(x, y) + direction

    # x: 1..15
    # y: 1..11
    def move(self, x, y):
        a = self._move_action(x, y)
        return self._maybe_render(a)

    def shoot(self, x, y):
        a = self._shoot_action(x, y)
        return self._maybe_render(a)

    def melee(self, x, y, direction):
        a = self._melee_action(x, y, direction)
        return self._maybe_render(a)

    def defend(self):
        # XXX: using slot to ensure 2-hex stacks are present with their "latest" hex
        stacks = {}  # {slot => (i, queuepos)}

        for y in range(11):
            for x in range(15):
                i = x * N_HEX_ATTRS
                hex = self.obs[0][y][i:i+N_HEX_ATTRS]
                side = hex[12]

                if side > 0 and side < 1:
                    slot = hex[13]
                    if side < 1:
                        slot += 7
                    queue = hex[11]
                    stacks[slot] = ((x, y), queue)

        # Find the "i" of the stack with the lowest QueuePos
        _, ((x, y), _) = min(stacks.items(), key=lambda x: x[1][1])
        return self.move(x+1, y+1)
