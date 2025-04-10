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

from .battlefield import Battlefield
from .value import Value
from .stats import GlobalStats, PlayerStats
from .hex import Hex, HexStateFlags, HexActionFlags
from .stack import Stack, StackFlags1, StackFlags2
from .. import pyprocconnector as pyconnector

NA = pyconnector.STATE_VALUE_NA


class Decoder:
    # declare as class var to allow overwriting by subclasses
    STATE_SIZE_ONE_HEX = pyconnector.STATE_SIZE_ONE_HEX

    # XXX: `state0` is used in autoencoder testing scenarios where it
    #      represents the real state, while `state` is the reconstructed one
    @classmethod
    def decode(cls, state, only_global=False, verbose=False):
        obs = state
        assert obs.shape == (pyconnector.STATE_SIZE,), f"{obs.shape} == ({pyconnector.STATE_SIZE},)"

        sizes = [
            pyconnector.STATE_SIZE_GLOBAL,
            pyconnector.STATE_SIZE_ONE_PLAYER,
            pyconnector.STATE_SIZE_ONE_PLAYER,
            pyconnector.STATE_SIZE_ONE_HEX * 11 * 15
        ]

        assert sum(sizes) == pyconnector.STATE_SIZE, f"{sum(sizes)} == {pyconnector.STATE_SIZE}"

        # calculate the indexes of the delimiters from the sizes
        delimiters = np.cumsum(sizes)[:-1]

        gstats, lstats, rstats, hexes = np.split(obs, delimiters)

        res = Battlefield(state)
        res.global_stats = cls.decode_global(gstats, verbose)
        res.left_stats = cls.decode_player(lstats, verbose)
        res.right_stats = cls.decode_player(rstats, verbose)

        if only_global:
            res.hexes = None
            res.stacks = None
            return res

        hexes = hexes.reshape(11, 15, pyconnector.STATE_SIZE_ONE_HEX)
        for y in range(11):
            row = []
            for x in range(15):
                hexdata = hexes[y][x]
                hex, stack = cls.decode_hex(hexdata, verbose)
                row.append(hex)
                if stack and not hex.IS_REAR.v:
                    res.stacks[stack.SIDE.v].append(stack)
            res.hexes.append(row)

        return res

    @classmethod
    def decode_global(cls, globaldata, verbose):
        res = {}
        for attr, (enctype, offset, n, vmax) in pyconnector.GLOBAL_ATTR_MAP.items():
            attrdata = globaldata[offset:][:n]
            res[attr] = Value(
                name=attr,
                enctype=enctype,
                n=n,
                vmax=vmax,
                raw=attrdata,
                verbose=verbose
            )

        return GlobalStats(**res, data=globaldata)

    @classmethod
    def decode_player(cls, playerdata, verbose):
        res = {}
        for attr, (enctype, offset, n, vmax) in pyconnector.PLAYER_ATTR_MAP.items():
            attrdata = playerdata[offset:][:n]
            res[attr] = Value(
                name=attr,
                enctype=enctype,
                n=n,
                vmax=vmax,
                raw=attrdata,
                verbose=verbose
            )

        return PlayerStats(**res, data=playerdata)

    @classmethod
    def decode_hex(cls, hexdata, verbose=False):
        res = {}
        sres = {}

        for attr, (enctype, offset, n, vmax) in pyconnector.HEX_ATTR_MAP.items():
            attrdata = hexdata[offset:][:n]
            kwargs = dict(
                name=attr,
                enctype=enctype,
                n=n,
                vmax=vmax,
                raw=attrdata,
                verbose=verbose,
            )

            if attr.startswith("STACK_"):
                sattr = attr.removeprefix("STACK_")
                if sattr == "FLAGS1":
                    sres[sattr] = Value(**dict(kwargs, struct_cls=StackFlags1, struct_mapping=pyconnector.STACK_FLAG1_MAP))
                elif sattr == "FLAGS2":
                    sres[sattr] = Value(**dict(kwargs, struct_cls=StackFlags2, struct_mapping=pyconnector.STACK_FLAG2_MAP))
                else:
                    sres[sattr] = Value(**kwargs)
            elif attr == "STATE_MASK":
                res[attr] = Value(**dict(kwargs, struct_cls=HexStateFlags, struct_mapping=pyconnector.HEX_STATE_MAP))
            elif attr == "ACTION_MASK":
                res[attr] = Value(**dict(kwargs, struct_cls=HexActionFlags, struct_mapping=pyconnector.HEX_ACT_MAP))
            else:
                res[attr] = Value(**kwargs)

        stack0 = Stack(hex=None, **dict(sres)) if sres["SIDE"].v is not None else None
        hex = Hex(**dict(res, data=hexdata, stack=None))

        # NOTE: no circular dependency (hex.stack.hex will be None and vice versa)
        # "shallow" links are still quite useful though
        stack = stack0._replace(hex=hex) if stack0 else None

        return hex._replace(stack=stack0), stack
