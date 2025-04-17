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

from ..pyprocconnector import (
    HEX_STATE_MAP,
    N_NONHEX_ACTIONS
)

from .hex import HEX_ACT_MAP


class Battlefield():
    def __repr__(self):
        return "Battlefield(11x15)"

    def __init__(self, envstate):
        self.hexes = []
        self.global_stats = None
        self.left_stats = None
        self.right_stats = None
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

    @property
    def my_stats(self):
        if self.global_stats.BATTLE_SIDE.v:
            return self.right_stats
        else:
            return self.left_stats

    @property
    def enemy_stats(self):
        if self.global_stats.BATTLE_SIDE.v:
            return self.left_stats
        else:
            return self.right_stats

    @property
    def is_battle_won(self):
        if self.global_stats.BATTLE_WINNER.v is None:
            return None

        return self.global_stats.BATTLE_WINNER.v == self.global_stats.BATTLE_SIDE.v

    # XXX: code below is for v7 and needs changes for v8

    # def get_lstack(self, i):
    #     return self._get_stack(0, i)

    # def get_rstack(self, i):
    #     return self._get_stack(1, i)

    # def _get_stack(self, side, i):
    #     if i >= 0 and i < len(self.stacks[side]):
    #         return self.stacks[side][i]
    #     else:
    #         print("Invalid stack (ID=%s)" % i)

    # XXX: this is C++ code translated in Python and it's quite ugly
    def render_battlefield(self):
        nocol = "\033[0m"
        redcol = "\033[31m"  # red
        bluecol = "\033[34m"  # blue
        darkcol = "\033[90m"
        activemod = "\033[107m\033[7m"  # bold+reversed
        ukncol = "\033[7m"  # white

        stacks = [[], []]  # left, right
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

        tablestartrow = len(lines)

        lines.append("    ₀▏₁▏₂▏₃▏₄▏₅▏₆▏₇▏₈▏₉▏₀▏₁▏₂▏₃▏₄")
        lines.append(" ┃▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┃ ")

        nummap = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
        addspace = True
        divlines = True

        # y even "▏"
        # y odd "▕"

        for y in range(11):
            for x in range(15):
                sym = "?"
                hex = self.get_hex(y, x)

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

                if hex.stack:
                    sym = hex.stack.alias()
                    if hex.stack.SIDE.v is None:
                        col = ukncol
                    elif hex.stack.SIDE.v:
                        col = bluecol
                    else:
                        col = redcol

                    if not hex.IS_REAR:
                        stacks[hex.stack.SIDE.v].append(hex.stack)
                    if hex.stack.QUANTITY.v is None:
                        col += "\033[1;47"
                    elif hex.stack.QUEUE.v[0] == 1:
                        col += activemod
                        assert hex.stack.SIDE.v == self.global_stats.BATTLE_SIDE_ACTIVE_PLAYER.v

                    if hex.stack.FLAGS1.v is not None and hex.stack.FLAGS1.struct.IS_WIDE:
                        if hex.stack.SIDE.v == 0 and hex.IS_REAR.v:
                            sym += "↠"
                            addspace = False
                        elif hex.stack.SIDE.v == 1 and not hex.IS_REAR.v and hex.X_COORD.v < 14:
                            sym += "↞"
                            addspace = False

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

        return lines, tablestartrow

    def render_side_stats(self, last_action, lines, tablestartrow):
        nocol = "\033[0m"
        redcol = "\033[31m"  # red
        bluecol = "\033[34m"  # blue

        #
        #  3. Add side table stuff
        #
        #     ₀▏₁▏₂▏₃▏₄▏₅▏₆▏₇▏₈▏₉▏₀▏₁▏₂▏₃▏₄
        #  ┃▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┃            Player: BLUE
        # ₀┨  ◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌ ┠₀           Action: SHOOT (y=4 x=5)
        # ₁┨ ◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌  ┠₁        DMG dealt: 0 (20988 since start)
        # ₂┨  ◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌▏◌ ┠₂     DMG received: 0 (7178 since start)
        # ₃┨ ◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕◌▕○▕○▕○  ┠₃     Value killed: 0 (1686820 since start)
        #  ...

        gstats = self.global_stats
        lpstats = self.left_stats
        rpstats = self.right_stats

        ended = gstats.BATTLE_WINNER.v is not None
        side = gstats.BATTLE_SIDE.v
        mystats = rpstats if side else rpstats

        for i, line in enumerate(lines):
            name = ""
            value = ""

            match i:
                case 1:
                    name = "Player"
                    aside = self.global_stats.BATTLE_SIDE_ACTIVE_PLAYER.v
                    if aside is not None:
                        value = (bluecol + "BLUE" + nocol) if aside else (redcol + "RED" + nocol)
                case 2:
                    name = "Action"
                    if last_action:
                        assert N_NONHEX_ACTIONS == 2
                        value = "%d: " % last_action
                        if last_action < 2:
                            value += "Wait" if last_action else "Retreat"
                        else:
                            hex = self.get_hex((last_action - 2) // len(HEX_ACT_MAP))
                            act = list(HEX_ACT_MAP)[(last_action - 2) % len(HEX_ACT_MAP)]
                            value += "%s (y=%s x=%s)" % (act, hex.Y_COORD.v, hex.X_COORD.v)
                case 3:
                    name = "DMG dealt"
                    value = "%d (%d since start)" % (mystats.DMG_DEALT_NOW_ABS.v, mystats.DMG_DEALT_ACC_ABS.v)
                case 4:
                    name = "DMG received"
                    value = "%d (%d since start)" % (mystats.DMG_RECEIVED_NOW_ABS.v, mystats.DMG_RECEIVED_ACC_ABS.v)
                case 5:
                    name = "Value killed"
                    value = "%d (%d since start)" % (mystats.VALUE_KILLED_NOW_ABS.v, mystats.VALUE_KILLED_ACC_ABS.v)
                case 6:
                    name = "Value lost"
                    value = "%d (%d since start)" % (mystats.VALUE_LOST_NOW_ABS.v, mystats.VALUE_LOST_ACC_ABS.v)
                case 7:
                    # XXX: if there's a draw, this text will be incorrect
                    restext = (bluecol + "BLUE WINS") if gstats.BATTLE_WINNER.v else (redcol + "RED WINS")
                    name = "Battle result"
                    value = (restext + nocol) if ended else ""

                case 8:
                    name = "Army value (L)"
                    value = "%d (%.0f‰ of current BF value)" % (lpstats.ARMY_VALUE_NOW_ABS.v, lpstats.ARMY_VALUE_NOW_REL.v)
                case 9:
                    name = "Army value (R)"
                    value = "%d (%.0f‰ of current BF value)" % (rpstats.ARMY_VALUE_NOW_ABS.v, rpstats.ARMY_VALUE_NOW_REL.v)
                case 10:
                    name = "Current BF value"
                    value = "%d (%.0f‰ of starting BF value)" % (gstats.BFIELD_VALUE_NOW_ABS.v, gstats.BFIELD_VALUE_NOW_REL0.v)
                case _:
                    continue

            lines[tablestartrow + i] += ("%s: %s" % (name.rjust(17), value))

        lines.append("")

    def render_stacks_table(self, lines):
        nocol = "\033[0m"
        redcol = "\033[31m"  # red
        bluecol = "\033[34m"  # blue
        activemod = "\033[107m\033[7m"  # bold+reversed

        arr = lambda x, n: [x for _ in range(n)]
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

        # All cell text is aligned right (5=default col width)
        colwidths = arr(5, 24)
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
            ("QUEUE", "Queue"),
            ("VALUE", "Value"),
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

            # Stack cols
            num_stacks = 10
            for side in [0, 1]:
                self.stacks[side].sort(key=lambda s: s.alias())
                slotstacks = [None] * num_stacks
                specialslot = 7
                for s in self.stacks[side]:
                    if s.alias() in [str(slot) for slot in range(7)]:
                        slotstacks[int(s.alias())] = s
                    elif specialslot < num_stacks:
                        assert s.alias() in ["M", "S"], s.alias()
                        slotstacks[specialslot] = s
                        specialslot += 1

                for i in range(num_stacks):
                    stack = slotstacks[i]
                    color = nocol
                    value = ""

                    if stack:
                        flags1 = stack.FLAGS1.struct if stack.FLAGS1.v is not None else None
                        flags2 = stack.FLAGS2.struct if stack.FLAGS2.v is not None else None
                        color = bluecol if side else redcol

                        if a == "ID":
                            value = stack.alias()
                        elif a == "VALUE":
                            if stack.VALUE_ONE.v is None:
                                value = "?"
                            elif stack.VALUE_ONE.v < 1000:
                                value = str(stack.VALUE_ONE.v)
                            elif stack.VALUE_ONE.v >= 1000:
                                value = "%.1f" % (stack.VALUE_ONE.v / 1000.0)
                                value = value[:-2] + "k" + value[-1]
                                if value.endswith("k0"):
                                    value = value[:-1]
                        elif a == "STATE":
                            if flags1 is None:
                                value = "?"
                            else:
                                value = ""
                                value += "W" if flags1.CAN_WAIT else " "
                                value += "A" if flags1.WILL_ACT else " "
                                value += "R" if flags1.CAN_RETALIATE else " "
                        elif a == "QUEUE":
                            value = str(stack.QUEUE.v.argmax())  # gets pos of first "1""
                        elif a == "ATTACK MODS":
                            if flags1 is None:
                                value = "?"
                            else:
                                value = ""
                                value += "D" if flags1 and flags1.ADDITIONAL_ATTACK else " "
                                value += "B" if flags2 and (flags2.BLIND_ATTACK or flags2.PETRIFY_ATTACK) else " "
                        elif a == "BLOCKED/ING":
                            if flags1 is None:
                                value = "?"
                            else:
                                value = "%s/%s" % (flags1.BLOCKED, flags1.BLOCKING)
                        elif a == "FLY/SLEEP":
                            if flags1 is None:
                                value = "?"
                            else:
                                value = "%s/%s" % (flags1.FLYING, flags1.SLEEPING)
                        elif a == "NO RETAL/MELEE":
                            if flags1 is None:
                                value = "?"
                            else:
                                value = "%s/%s" % (flags1.BLOCKS_RETALIATION, flags1.NO_MELEE_PENALTY)
                        elif a == "WIDE/BREATH":
                            if flags1 is None:
                                value = "?"
                            else:
                                value = "%s/%s" % (flags1.IS_WIDE, flags1.TWO_HEX_ATTACK_BREATH)
                        elif a == "Y_COORD":
                            value = str(stack.hex.Y_COORD.v)
                        elif a == "X_COORD":
                            value = str(stack.hex.X_COORD.v)
                        else:
                            value = str(getattr(stack, a).v)

                        if stack.QUEUE.v[0] == 1 and self.global_stats.BATTLE_WINNER.v is None:
                            color += activemod

                    colid = 2 + i + side + (num_stacks*side)
                    row[colid] = (color, colwidths[colid], value)

            if a == "Y_COORD":
                specialcounter += 1

            table.append(row)

        for r in table:
            line = ""
            for color, width, txt in r:
                line = line + color + txt.rjust(width) + nocol

            lines.append(line)

    def render(self, last_action):
        lines, tablestartrow = self.render_battlefield()
        self.render_side_stats(last_action, lines, tablestartrow)
        self.render_stacks_table(lines)
        return "\n".join(lines)
