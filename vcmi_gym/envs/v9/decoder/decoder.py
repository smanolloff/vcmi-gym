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
from collections import namedtuple

from .battlefield import Battlefield
from .value import Value
from .stats import GlobalStats, PlayerStats
from .hex import Hex, HexStateFlags, HexActionFlags
from .link import Link
# from .stack import Stack, StackFlags
from .. import pyprocconnector as pyconnector

NA = pyconnector.STATE_VALUE_NA


class StackFlags(namedtuple("StackFlags", list(pyconnector.STACK_FLAG_MAP.keys()))):
    def __repr__(self):
        return "{%s}" % ", ".join([f for f in self._fields if getattr(self, f)])


class Decoder:
    # declare as class var to allow overwriting by subclasses
    STATE_SIZE_ONE_HEX = pyconnector.STATE_SIZE_ONE_HEX

    # XXX: `state0` is used in autoencoder testing scenarios where it
    #      represents the real state, while `state` is the reconstructed one
    @classmethod
    def decode(cls, pyresult, only_global=False, verbose=False):
        obs = pyresult.state

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

        res = Battlefield(pyresult)
        res.global_stats = cls.decode_global(gstats, verbose)
        res.left_stats = cls.decode_player(lstats, verbose)
        res.right_stats = cls.decode_player(rstats, verbose)

        if only_global:
            return res

        hexes = hexes.reshape(11, 15, pyconnector.STATE_SIZE_ONE_HEX)
        for y in range(11):
            row = []
            for x in range(15):
                hexdata = hexes[y][x]
                row.append(cls.decode_hex(hexdata, verbose))
            res.hexes.append(row)

        linktypes = pyresult.edge_types.reshape(pyresult.num_edges, -1)
        flathexes = [h for row in res.hexes for h in row]
        for (src, dst, v, type) in zip(pyresult.edge_sources, pyresult.edge_targets, pyresult.edge_values, linktypes):
            res.links.append(cls.decode_link(flathexes[src], flathexes[dst], v, type))

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

    #
    # XXX: not adjusted for v9
    #

    # @classmethod
    # def decode_stack(cls, stackdata, stackdata0=None, precision=None, roundfracs=None, verbose=False):
    #     res = {}
    #     for attr, (enctype, offset, n, vmax) in pyconnector.STACK_ATTR_MAP.items():
    #         attrdata = stackdata[offset:][:n]
    #         attrdata0 = stackdata0[offset:][:n] if stackdata0 is not None else None
    #         kwargs = dict(
    #             name=attr,
    #             enctype=enctype,
    #             n=n,
    #             vmax=vmax,
    #             raw=attrdata,
    #             raw0=attrdata0,
    #             precision=precision,    # number of digits after "."
    #             roundfracs=roundfracs,  # 5 = round to nearest 0.2 (3.14 => 3.2)
    #             verbose=verbose
    #         )

    #         if attr == "FLAGS":
    #             res[attr] = Value(**dict(kwargs, struct_cls=StackFlags, struct_mapping=pyconnector.STACK_FLAG_MAP))
    #         else:
    #             res[attr] = Value(**kwargs)

    #     return Stack(**dict(res, data=stackdata))

    @classmethod
    def decode_hex(cls, hexdata, verbose=False):
        res = {}
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

            if attr == "STATE_MASK":
                res[attr] = Value(**dict(kwargs, struct_cls=HexStateFlags, struct_mapping=pyconnector.HEX_STATE_MAP))
            elif attr == "ACTION_MASK":
                res[attr] = Value(**dict(kwargs, struct_cls=HexActionFlags, struct_mapping=pyconnector.HEX_ACT_MAP))
            elif attr == "STACK_FLAGS":
                res[attr] = Value(**dict(kwargs, struct_cls=StackFlags, struct_mapping=pyconnector.STACK_FLAG_MAP))
            else:
                res[attr] = Value(**kwargs)
        hex = Hex(**dict(res, data=hexdata, links=dict(inbound=[], outbound=[])))
        return hex

    @classmethod
    def decode_link(cls, src, dst, v, type, verbose=False):
        link = Link(
            src=src,
            dst=dst,
            value=v,
            type=list(pyconnector.LINK_TYPE_MAP.keys())[type.argmax()]
        )

        link.src.links["outbound"].append(link)
        link.dst.links["inbound"].append(link)

        return link
