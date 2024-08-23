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

import types
import ctypes
import os
from collections import OrderedDict

if (os.getenv("VCMIGYM_DEBUG", None) == "1"):
    from ...connectors.build import exporter_v2
else:
    from ...connectors.rel import exporter_v2

# Re-exported vars
from ..v1.pyconnector import (  # noqa: F401
    N_NONHEX_ACTIONS,
    N_HEX_ACTIONS,
    N_ACTIONS,
    STATE_VALUE_NA,
    HEXACTMAP,
    HEXSTATEMAP,
    DMGMODMAP,
    SHOOTDISTMAP,
    MELEEDISTMAP,
    SIDEMAP,
    PyActmask,
    PyAttnmask,
    PyConnector as PyConnector_v1,
)

EXPORTER = exporter_v2.Exporter()
STATE_SIZE = EXPORTER.get_state_size()
STATE_SIZE_ONE_HEX = EXPORTER.get_state_size_one_hex()
ATTRMAP = types.MappingProxyType(OrderedDict([(k, tuple(v)) for k, *v in EXPORTER.get_attribute_mapping()]))
PyState = ctypes.c_float * STATE_SIZE


class PyConnector(PyConnector_v1):
    class PyRawState(ctypes.Structure):
        _fields_ = PyConnector_v1.PyRawState._fields_.copy()
        _fields_[0] = ("state", PyState)

    def _get_connector(self):
        from ...connectors.rel import connector_v2
        return connector_v2
