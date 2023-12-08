from .pyconnector import (
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
    N_HEX_ATTRS,
    N_STACK_ATTRS
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

        for i in range(165):
            # i_qty = state index of the Qty stack attribute at i-th hex
            i0 = i*N_HEX_ATTRS + (N_HEX_ATTRS - N_STACK_ATTRS)
            is_enemy = self.obs[i0+11]

            if is_enemy > 0 and is_enemy < 1:
                slot = self.obs[i0+12]
                queue = self.obs[i0+10]
                stacks[slot] = (i, queue)

        # Find the "i" of the stack with the lowest QueuePos
        _, (mi, _) = min(stacks.items(), key=lambda x: x[1][1])
        return self.move(1 + mi % 15, 1 + mi // 15)
