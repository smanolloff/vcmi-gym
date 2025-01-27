# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np

from .stack import Stack, StackFlags
from .value import Value

from ..pyprocconnector import (
    HEX_STATE_MAP,
    HEX_ATTR_MAP,
    STATE_SIZE_MISC,
    STATE_SIZE_STACKS,
    STATE_SIZE_HEXES,
    STATE_SIZE_ONE_STACK,
    STATE_SIZE_ONE_HEX,
    STACK_ATTR_MAP,
    STACK_FLAG_MAP
)


class Battlefield():
    def __repr__(self):
        return "Battlefield(11x15)"

    def __init__(self, is_battle_over, envstate=None):
        self.is_battle_over = is_battle_over
        self.hexes = []
        self.stacks = [[], []]  # left, right stacks
        self.envstate = envstate

    def get_hex(self, y_or_n, x=None):
        if x is not None:
            y = y_or_n
        else:
            y = y_or_n // 15
            x = y_or_n % 15

        if y >= 0 and y < len(self.hexes) and x >= 0 and x < len(self.hexes[y]):
            return self.hexes[y][x]
        else:
            print("Invalid hex (y=%s x=%s)" % (y, x))

    def get_lstack(self, i):
        return self._get_stack(0, i)

    def get_rstack(self, i):
        return self._get_stack(1, i)

    def _get_stack(self, side, i):
        if i >= 0 and i < len(self.stacks[side]):
            return self.stacks[side][i]
        else:
            print("Invalid stack (ID=%s)" % i)

    # XXX: this is C++ code translated in Python and it's quite ugly
    def render(self, strict=False):
        astack = None

        arr = lambda x, n: [x for _ in range(n)]
        idstacks = [arr(None, 10), arr(None, 10)]

        for sidestacks in self.stacks:
            for stack in sidestacks:
                if stack.exists():
                    idstacks[stack.SIDE.v][stack.ID.v] = stack
                    if stack.QUEUE_POS.v == 0:
                        astack = stack

        if not astack:
            # raise Exception("Could not find active stack")
            print("WARNING: Could not find active stack")

        nocol = "\033[0m"
        redcol = "\033[31m"  # red
        bluecol = "\033[34m"  # blue
        darkcol = "\033[90m"
        activemod = "\033[107m\033[7m"  # bold+reversed
        # ukncol = "\033[7m"  # white

        lines = []

        #
        # 2. Build ASCII table
        #    (+populate aliveStacks var)
        #    NOTE: the contents below look mis-aligned in some editors.
        #          In (my) terminal, it all looks correct though.
        #
        #   ▕₁▕₂▕₃▕₄▕₅▕₆▕₇▕₈▕₉▕₀▕₁▕₂▕₃▕₄▕₅▕
        #  ┃▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┃
        # ¹┨  1 ◌ ○ ○ ○ ○ ◌ ◌ ◌ ◌ ◌ ◌ ◌ ◌ 1 ┠¹
        # ²┨ ◌ ○ ○ ○ ○ ○ ○ ◌ ◌ ◌ ◌ ◌ ◌ ◌ ◌  ┠²
        # ³┨  ◌ ○ ○ ○ ○ ○ ○ ◌ ▦ ▦ ◌ ◌ ◌ ◌ ◌ ┠³
        # ⁴┨ ◌ ○ ○ ○ ○ ○ ○ ○ ▦ ▦ ▦ ◌ ◌ ◌ ◌  ┠⁴
        # ⁵┨  2 ◌ ○ ○ ▦ ▦ ◌ ○ ◌ ◌ ◌ ◌ ◌ ◌ 2 ┠⁵
        # ⁶┨ ◌ ○ ○ ○ ▦ ▦ ◌ ○ ○ ◌ ◌ ◌ ◌ ◌ ◌  ┠⁶
        # ⁷┨  3 3 ○ ○ ○ ▦ ◌ ○ ○ ◌ ◌ ▦ ◌ ◌ 3 ┠⁷
        # ⁸┨ ◌ ○ ○ ○ ○ ○ ○ ○ ○ ◌ ◌ ▦ ▦ ◌ ◌  ┠⁸
        # ⁹┨  ◌ ○ ○ ○ ○ ○ ○ ○ ◌ ◌ ◌ ◌ ◌ ◌ ◌ ┠⁹
        # ⁰┨ ◌ ○ ○ ○ ○ ○ ○ ○ ◌ ◌ ◌ ◌ ◌ ◌ ◌  ┠⁰
        # ¹┨  4 ◌ ○ ○ ○ ○ ○ ◌ ◌ ◌ ◌ ◌ ◌ ◌ 4 ┠¹
        #  ┃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁┃
        #   ▕¹▕²▕³▕⁴▕⁵▕⁶▕⁷▕⁸▕⁹▕⁰▕¹▕²▕³▕⁴▕⁵▕
        #

        lines.append("    ₀▏₁▏₂▏₃▏₄▏₅▏₆▏₇▏₈▏₉▏₀▏₁▏₂▏₃▏₄")
        lines.append(" ┃▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┃ ")

        nummap = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
        addspace = True
        seenstacks = [arr([], 10), arr([], 10)]
        divlines = True

        # y even "▏"
        # y odd "▕"

        def uknstack(id, y, x, side):
            attrs = {k: Value(k, "BINARY_EXPLICIT_NULL", 2, 1, np.array([1, 0])) for k in STACK_ATTR_MAP.keys()}
            attrs["COORD_Y"] = Value("COORD_Y", "BINARY_EXPLICIT_NULL", len(STACK_FLAG_MAP), 10, np.array([0]*len(STACK_FLAG_MAP)), struct_cls=StackFlags, struct_mapping=STACK_FLAG_MAP)
            attrs["FLAGS"] = Value("FLAGS", "BINARY_EXPLICIT_NULL", len(STACK_FLAG_MAP), 1, np.array([0]*len(STACK_FLAG_MAP)), struct_cls=StackFlags, struct_mapping=STACK_FLAG_MAP)
            return Stack(**dict(attrs, data=[]))

        for y in range(11):
            for x in range(15):
                sym = "?"
                hex = self.get_hex(y, x)
                stack = None

                if hex.STACK_ID.v is not None:
                    stack = idstacks[hex.STACK_SIDE.v][hex.STACK_ID.v]
                    if not stack:
                        assert not strict, "stack with side=%d and ID=%d not found" % (hex.STACK_SIDE.v, hex.STACK_ID.v)
                        stack = uknstack(hex.STACK_ID.v, hex.Y_COORD.v, hex.X_COORD.v, hex.STACK_SIDE.v)

                if x == 0:
                    lines.append("%s┨%s" % (nummap[y % 10], " " if y % 2 == 0 else ""))

                # XXX: must REPLACE lines[-1] = row in the end as in python
                #      string are immutable
                row = lines[-1]

                if addspace:
                    if divlines and x != 0:
                        row += darkcol
                        row += "▏" if y % 2 == 0 else "▕"
                    else:
                        row += " "

                addspace = True
                smask = hex.STATE_MASK.v
                col = nocol

                # First put symbols based on hex state.
                # If there's a stack on this hex, symbol will be overriden.
                mpass = HEX_STATE_MAP["PASSABLE"]
                mstop = HEX_STATE_MAP["STOPPING"]
                mdmgl = HEX_STATE_MAP["DAMAGING_L"]
                mdmgr = HEX_STATE_MAP["DAMAGING_R"]

                symbols = [
                    ("⨻", bluecol, [mpass, mstop, mdmgl]),
                    ("⨻", redcol, [mpass, mstop, mdmgr]),
                    ("✶", bluecol, [mpass, mdmgl]),
                    ("✶", redcol, [mpass, mdmgr]),
                    ("△", nocol, [mpass, mstop]),
                    ("○", nocol, [mpass]),  # changed to "◌" if unreachable
                    ("◼", nocol, [])
                ]

                for s, c, m in symbols:
                    if all(smask[m]):
                        sym = s
                        col = c
                        break

                hexactions = hex.actions()
                if col == nocol and "MOVE" not in hexactions:
                    col = darkcol
                    sym = "◌" if sym == "○" else sym

                if hex.STACK_ID.v is not None:
                    seen = seenstacks[stack.SIDE.v][stack.ID.v]
                    sym = stack.alias()
                    col = bluecol if stack.SIDE.v else redcol

                    if stack.QUANTITY.v == -1:
                        col += "\033[1;47"
                    elif stack.QUEUE_POS.v == 0:
                        col += activemod

                    if stack.FLAGS.struct.IS_WIDE and not seen:
                        if stack.SIDE.v == 0:
                            sym += "↠"
                            addspace = False
                        elif stack.SIDE.v == 1 and hex.X_COORD.v < 14:
                            sym += "↞"
                            addspace = False

                    seenstacks[stack.SIDE.v][stack.ID.v] = 1

                row += (col + sym + nocol)

                if x == 14:
                    row += (" " if y % 2 == 0 else "  ")
                    row += "┠"
                    row += nummap[y % 10]

                # XXX: need to replace as strings are immutable in python
                lines[-1] = row

        lines.append(" ┃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁┃")
        lines.append("   ⁰▕¹▕²▕³▕⁴▕⁵▕⁶▕⁷▕⁸▕⁹▕⁰▕¹▕²▕³▕⁴")

        lines.append("")

        #
        # 5. Add stacks table:
        #
        #          Stack # |   0   1   2   3   4   5   6   A   B   C   0   1   2   3   4   5   6   A   B   C
        # -----------------+--------------------------------------------------------------------------------
        #              Qty |   0  34   0   0   0   0   0   0   0   0   0  17   0   0   0   0   0   0   0   0
        #           Attack |   0   8   0   0   0   0   0   0   0   0   0   6   0   0   0   0   0   0   0   0
        #    ...10 more... | ...
        # -----------------+--------------------------------------------------------------------------------
        #
        # table with 24 columns (1 header, 3 dividers, 10 stacks per side)
        # Each row represents a separate attribute

        # using RowDef = std::tuple<StackAttribute, std::string>;

        # All cell text is aligned right (4=default col width)
        colwidths = arr(4, 24)
        colwidths[0] = 16  # header col

        # Divider column indexes
        divcolids = (1, 12, 23)

        for i in divcolids:
            colwidths[i] = 2  # divider col

        # {Attribute, name, colwidth}
        rowdefs = [
            ("ID", "Stack #"),
            ("DIVROW", ""),  # divider row
            ("QUANTITY", "Qty"),
            ("ATTACK", "Attack"),
            ("DEFENSE", "Defense"),
            ("SHOTS", "Shots"),
            ("DMG_MIN", "Dmg (min)"),
            ("DMG_MAX", "Dmg (max)"),
            ("HP", "HP"),
            ("HP_LEFT", "HP left"),
            ("SPEED", "Speed"),
            ("QUEUE_POS", "Queue"),
            ("AI_VALUE", "Value"),
            ("STATE", "State"),  # "WAR" = CAN_WAIT, WILL_ACT, CAN_RETAL
            ("ATTACK MODS", "Attack mods"),  # "DB" = Double, Blinding
            ("BLOCKED/ING", "Blocked/ing"),
            ("FLY/SLEEP", "Fly/Sleep"),
            ("NO RETAL/MELEE", "No Retal/Melee"),
            ("WIDE/BREATH", "Wide/Breath"),
            ("Y_COORD", "Y"),
            ("X_COORD", "X"),
            # ("STACK_ID_NULL", "STACK_ID_NULL"),
            # ("ID_NULL", "ID_NULL"),
            # ("STACK_SIDE_NULL", "STACK_SIDE_NULL"),
            # ("Y_COORD_NULL", "Y_COORD_NULL"),
            # ("X_COORD_NULL", "X_COORD_NULL"),
            ("DIVROW", ""),  # divider row
        ]

        # Table with nrows and ncells, each cell a 3-element tuple
        # cell: color, width, txt
        # using TableCell = std::tuple<std::string, int, std::string>;
        # using TableRow = std::array<TableCell, colwidths.size()>;

        def makerow():
            return arr(("", 0, ""), len(colwidths))

        table = []
        divrow = makerow()

        for i in range(len(colwidths)):
            divrow[i] = (nocol, colwidths[i], "-" * colwidths[i])

        for i in divcolids:
            divrow[i] = (nocol, colwidths[i], "-" * (colwidths[i]-1) + "+")

        specialcounter = 0

        # Attribute rows
        for a, aname in rowdefs:
            if a == "DIVROW":
                table.append(divrow)
                continue

            row = makerow()

            # Header col
            row[0] = (nocol, colwidths[0], aname)

            # Div cols
            for i in [1, 12, len(colwidths)-1]:
                row[i] = (nocol, colwidths[i], "|")

            def get_stack_null_value(i, attrname):
                enctype, offset, *_ = STACK_ATTR_MAP[attrname]
                assert enctype.endswith("EXPLICIT_NULL")
                stacksdata = self.envstate[STATE_SIZE_MISC:][:STATE_SIZE_STACKS]
                stackdata = stacksdata[i*STATE_SIZE_ONE_STACK:][:STATE_SIZE_ONE_STACK]
                attrdata = stackdata[offset:]
                return attrdata[0]

            def get_hex_null_value(i, attrname):
                enctype, offset, *_ = HEX_ATTR_MAP[attrname]
                assert enctype.endswith("EXPLICIT_NULL")
                hexesdata = self.envstate[STATE_SIZE_MISC+STATE_SIZE_STACKS:]
                assert hexesdata.shape[0] == STATE_SIZE_HEXES
                hexdata = hexesdata[i*STATE_SIZE_ONE_HEX:][:STATE_SIZE_ONE_HEX]
                attrdata = hexdata[offset:]
                return attrdata[0]

            # Stack cols
            for side in [0, 1]:
                sidestacks = self.stacks[side]
                for i in range(len(sidestacks)):
                    stack = sidestacks[i]
                    color = nocol
                    value = ""

                    if stack.exists():
                        flags = stack.FLAGS.struct
                        color = bluecol if stack.SIDE.v else redcol

                        if a == "ID":
                            value = stack.alias()
                        elif a == "AI_VALUE" and getattr(stack, a).v >= 1000:
                            value = "%.1f" % (getattr(stack, a).v / 1000.0)
                            value = value[:-2] + "k" + value[-1]
                            if value.endswith("k0"):
                                value = value[:-1]
                        elif a == "STATE":
                            value = ""
                            value += "W" if flags.CAN_WAIT else " "
                            value += "A" if flags.WILL_ACT else " "
                            value += "R" if flags.CAN_RETALIATE else " "
                        elif a == "ATTACK MODS":
                            value = ""
                            value += "D" if flags.ADDITIONAL_ATTACK else " "
                            value += "B" if flags.BLIND_LIKE_ATTACK else " "
                        elif a == "BLOCKED/ING":
                            value = "%s/%s" % (flags.BLOCKED, flags.BLOCKING)
                        elif a == "FLY/SLEEP":
                            value = "%s/%s" % (flags.FLYING, flags.SLEEPING)
                        elif a == "NO RETAL/MELEE":
                            value = "%s/%s" % (flags.BLOCKS_RETALIATION, flags.NO_MELEE_PENALTY)
                        elif a == "WIDE/BREATH":
                            value = "%s/%s" % (flags.IS_WIDE, flags.TWO_HEX_ATTACK_BREATH)

                        # not sure what I was after here, but with the extended
                        # logic in the "Value" objects it may be redunant

                        # elif a == "STACK_ID_NULL":
                        #     value = "%.1f" % get_hex_null_value(y, x, "STACK_ID")
                        # elif a == "ID_NULL":
                        #     value = "%.1f" % get_stack_null_value(i, side, "ID")
                        # elif a == "STACK_SIDE_NULL":
                        #     value = "%.1f" % get_stack_null_value(i, side, "SIDE")
                        # elif a == "Y_COORD_NULL":
                        #     value = "%.1f" % get_stack_null_value(i, side, "Y_COORD")
                        # elif a == "X_COORD_NULL":
                        #     value = "%.1f" % get_stack_null_value(i, side, "X_COORD")
                        else:
                            value = str(getattr(stack, a).v)

                        if stack.QUEUE_POS.v == 0 and not self.is_battle_over:
                            color += activemod

                    colid = 2 + i + side + (10*side)
                    row[colid] = (color, colwidths[colid], value)

            if a == "Y_COORD":
                specialcounter += 1

            table.append(row)

        for r in table:
            line = ""
            for color, width, txt in r:
                line = line + color + txt.rjust(width) + nocol

            lines.append(line)

        #
        # 7. Join rows into a single string
        #
        return "\n".join(lines)
