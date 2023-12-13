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
        stacks = {}  # {slot => (i, queuepos)}

        #
        # THE UGLIEST CODE
        #
        for y in range(11):
            for x in range(15):
                i = x * N_HEX_ATTRS
                hex = self.obs[0][y][i:i+N_HEX_ATTRS]
                side = hex[12]

                # >0 means stack is not N/A
                if side > 0:
                    slot = hex[13]

                    # 1 means RIGHT (defender)
                    if side == 1:
                        # offset defender slots to be outside 0..1 range
                        # (to prevent overwriting attacker slots)
                        slot -= 1

                    queue = hex[11]

                    # print("(%s,%s) slot: %s, queue: %s" % (x+1, y+1, slot, queue))

                    # If we have an entry for this slot => it's a 2-hex stack
                    # We sweep bfield left-to-right and we want to keep the
                    # rightmost hex for attacker => overwrite
                    # for defender, we want leftmost hex => don't overwrite
                    # XXX: the above will not work for BLUE
                    if slot in stacks:
                        # overwrite hex only for "left" side
                        if side < 1:
                            stacks[slot] = ((x, y), queue)
                    else:
                        stacks[slot] = ((x, y), queue)

        # Find the "i" of the stack with the lowest QueuePos
        _, ((x, y), _) = min(stacks.items(), key=lambda x: x[1][1])
        return self.move(x+1, y+1)
