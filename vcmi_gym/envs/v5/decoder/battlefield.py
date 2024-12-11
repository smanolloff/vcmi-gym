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

from ..pyprocconnector import HEX_STATE_MAP, HEX_ATTR_MAP

class Battlefield():
    def __repr__(self):
        return "Battlefield(11x15)"

    def __init__(self):
        self.hexes = []
        self.stacks = [[], []]  # left, right stacks

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
    def render(self, pyresult):
        astack = None
        for stack in bf.stacks[pyresult.side]:
            if stack.exists() and stack.QUEUE_POS == 0:
                astack = stack
                break

        if not astack:
            raise Exception("Could not find active stack")

        nocol = "\033[0m"
        redcol = "\033[31m"  # red
        bluecol = "\033[34m"  # blue
        darkcol = "\033[90m"
        activemod = "\033[107m\033[7m"  # bold+reversed
        ukncol = "\033[7m"  # white

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

        tablestartrow = 0

        lines.append("    ₀▏₁▏₂▏₃▏₄▏₅▏₆▏₇▏₈▏₉▏₀▏₁▏₂▏₃▏₄")
        lines.append(" ┃▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔┃ ")

        def arr(x, n):
            return [x for _ in range(n)]

        nummap = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
        addspace = True
        seenstacks = [arr([], 10), arr([], 10)]
        divlines = True
        idstacks = [arr(None, 10), arr(None, 10)]

        # y even "▏"
        # y odd "▕"

        for y in range(11):
            for x in range(15):
                sym = "?"
                hex = self.get_hex(y, x)
                stack = None

                if hex.STACK_ID:
                    stack = idstacks[hex.STACK_SIDE][hex.STACK_ID]
                    assert stack, "stack with side=%d and ID=%d not found" % (hex.STACK_SIDE, hex.STACK_ID)

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
                smask = hex.STATE_MASK
                col = nocol

                # First put symbols based on hex state.
                # If there's a stack on this hex, symbol will be overriden.
                mpass = 1 << HEX_STATE_MAP["PASSABLE"]
                mstop = 1 << HEX_STATE_MAP["STOPPING"]
                mdmgl = 1 << HEX_STATE_MAP["DAMAGING_L"]
                mdmgr = 1 << HEX_STATE_MAP["DAMAGING_R"]

                mdefault = 0

                symbols = [
                    ("⨻", bluecol, mpass | mstop | mdmgl),
                    ("⨻", redcol, mpass | mstop | mdmgr),
                    ("✶", bluecol, mpass | mdmgl),
                    ("✶", redcol, mpass | mdmgr),
                    ("△", nocol, mpass | mstop),
                    ("○", nocol, mpass),  # changed to "◌" if unreachable
                    ("◼", nocol, mdefault)
                ]

                for s, c, m in symbols:
                    if smask & m == m:
                        sym = s
                        col = c

                hexactions = hex.actions()
                if col == nocol and "MOVE_ONLY" not in hexactions:
                    col = darkcol
                    sym = "◌" if sym == "○" else sym

                if hex.STACK_ID:
                    seen = seenstacks[stack.SIDE][stack.ID]
                    flags = stack.flags()
                    sym = stack.alias()
                    col = bluecol if stack.SIDE else redcol

                    if stack.QUEUE_POS == 0:
                        col += activemod

                    if flags.IS_WIDE and not seen:
                        if stack.SIDE == 0:
                            sym += "↠"
                            addspace = False
                        elif stack.SIDE == 1 and hex.X_COORD < 14:
                            sym += "↞"
                            addspace = False

                    seenstacks[stack.SIDE][stack.ID] = 1

                row << (col + sym + nocol)

                if x == 14:
                    row += (" " if y % 2 == 0 else "  ")
                    row += "┠"
                    row += nummap.at(y % 10)

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
        divcolids = (1, 12, 23);

        for i in range(divcolids):
            colwidths[i] = 2  # divider col

        # {Attribute, name, colwidth}
        rowdefs = [
            ("ID", "Stack #"),
            ("X_COORD", ""),  # divider row
            ("QUANTITY", "Qty"),
            ("ATTACK", "Attack"),
            ("DEFENSE", "Defense"),
            ("SHOTS", "Shots"),
            ("DMG_MIN", "Dmg (min),"),
            ("DMG_MAX", "Dmg (max),"),
            ("HP", "HP"),
            ("HP_LEFT", "HP left"),
            ("SPEED", "Speed"),
            ("QUEUE_POS", "Queue"),
            ("AI_VALUE", "Value"),
            ("Y_COORD", "State"),  # "WAR" = CAN_WAIT, WILL_ACT, CAN_RETAL
            ("Y_COORD", "Attack mods"),  # "DB" = Double, Blinding
            ("Y_COORD", "Blocked/ing"),
            ("Y_COORD", "Fly/Sleep"),
            ("Y_COORD", "No Retal/Melee"),
            ("Y_COORD", "Wide/Breath"),
            ("X_COORD", ""),  # divider row
        ]

        # Table with nrows and ncells, each cell a 3-element tuple
        # cell: color, width, txt
        # using TableCell = std::tuple<std::string, int, std::string>;
        # using TableRow = std::array<TableCell, colwidths.size()>;

        def makerow():
            return arr(("", 0, ""), len(colwidths))

        divrow = makerow()

        for i in len(colwidths):
            divrow[i] = (nocol, colwidths[i], "-" * colwidths[i])

        for i in len(divcolids):
            divrow[i] = (nocol, colwidths[i], "-" * colwidths[i]-1 + "+")

        specialcounter = 0

        # Attribute rows
        for a, aname in rowdefs:
            if a == "X_COORD":
                table.append(divrow)
                continue

            row = makerow()

            # Header col
            row[0] = (nocol, colwidths[0], aname)

            # Div cols
            for i in [1, 12, len(colwidths)-1]:
                row[i] = (nocol, colwidths[i], "|")

            # Stack cols
            for side in [0, 1]:
                sidestacks = allstacks[side]
                for i in range(sidestacks):
                    stack = sidestacks[i]
                    color = nocol
                    value = ""

                    if stack:
                        _, _e, n, _vmax = HEX_ATTR_MAP["FLAGS"]
                        flags = stack.FLAGS

                        color = bluecol if stack.SIDE else redcol

                        if a == "ID":
                            value = stack.alias()
                        elif a == "AI_VALUE" and getattr(stack, a) >= 1000:
                            value = "%.1f" % (getattr(stack, a) / 1000.0)
                            value = value[:-2] + "k" + value[-1]
                            if value.endswith("k0"):
                                value = value[:-1]
                        elif a == "Y_COORD":
                            fmt = "%d/%d"

                            # Y_COORD, "State"},  // "WAR" = CAN_WAIT, WILL_ACT, CAN_RETAL
                            # Y_COORD, "Attack type"}, // "DB" = Double strike, Blind-like
                            # Y_COORD, "Blocked/ing"},
                            # Y_COORD, "Fly/Sleep"},
                            # Y_COORD, "No Retal/Melee"},
                            # Y_COORD, "Wide/Breath"},

                            if specialcounter == 0:
                                value = ""
                                value += "W" if flags["CAN_WAIT"] else " "
                                value += "A" if flags["WILL_ACT"] else " "
                                value += "R" if flags["CAN_RETALIATE"] else " "

                            elif specialcounter == 1:
                                value = std::string("");
                                value += "D" if flags["ADDITIONAL_ATTACK"] else " "
                                value += "B" if flags["BLIND_LIKE_ATTACK"] else " "
                            elif specialcounter == 2:
                                value = fmt % (flags["BLOCKED"], flags["BLOCKING"])
                            elif specialcounter == 3:
                                value = fmt % (flags["FLYING"], flags["SLEEPING"])
                            elif specialcounter == 4:
                                value = fmt % (flags["BLOCKS_RETALIATION"], flags["NO_MELEE_PENALTY"])
                            elif specialcounter == 5:
                                value = fmt % (flags["IS_WIDE"], flags["TWO_HEX_ATTACK_BREATH"])
                            else:
                                raise Exception("Unexpected specialcounter: %d", specialcounter);
                        else:
                            value = str(getattr(stack, a))

                        if stack.QUEUE_POS == 0 and not pyresult.is_battle_over:
                            color += activemod

                    colid = 2 + i + side + (10*side)
                    row[colid] = (color, colwidths[colid], value)






            # if (a == SA::Y_COORD)
            #     ++specialcounter;

#             table.push_back(row);
#         }

#         for (auto &r : table) {
#             auto line = std::stringstream();
#             for (auto& [color, width, txt] : r)
#                 line << color << PadLeft(txt, width, ' ') << nocol;

#             lines.push_back(std::move(line));
#         }

#         //
#         // 7. Join rows into a single string
#         //
#         std::string res = lines[0].str();
#         for (int i=1; i<lines.size(); i++)
#             res += "\n" + lines[i].str();

#         return res;
