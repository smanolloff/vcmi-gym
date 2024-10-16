// =============================================================================
// Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "schema/v2/constants.h"
#include "exporter.h"

namespace Connector::V2 {
    const int Exporter::getVersion() const { return 2; }
    const int Exporter::getStateSize() const { return MMAI::Schema::V2::BATTLEFIELD_STATE_SIZE; }
    const int Exporter::getStateSizeOneHex() const { return MMAI::Schema::V2::BATTLEFIELD_STATE_SIZE_ONE_HEX; }

    const std::vector<V1::AttributeMapping> Exporter::getAttributeMapping() const {
        return _getAttributeMapping(MMAI::Schema::V2::HEX_ENCODING);
    }

    PYBIND11_MODULE(exporter_v2, m) {
        pybind11::class_<Exporter>(m, "Exporter")
            .def(pybind11::init<>())
            .def("get_version", &Exporter::getVersion)
            .def("get_n_actions", &Exporter::getNActions)
            .def("get_n_nonhex_actions", &Exporter::getNNonhexActions)
            .def("get_n_hex_actions", &Exporter::getNHexActions)
            .def("get_state_size", &Exporter::getStateSize)
            .def("get_state_size_one_hex", &Exporter::getStateSizeOneHex)
            .def("get_state_value_na", &Exporter::getStateValueNa)
            .def("get_side_left", &Exporter::getSideLeft)
            .def("get_side_right", &Exporter::getSideRight)
            .def("get_dmgmods", &Exporter::getDmgmods, "Get a list of the DmgMod enum value names")
            .def("get_shootdistances", &Exporter::getShootdistances, "Get a list of the ShootDistance enum value names")
            .def("get_meleedistances", &Exporter::getMeleedistances, "Get a list of the MeleeDistance enum value names")
            .def("get_hexactions", &Exporter::getHexactions, "Get a list of the HexAction enum value names")
            .def("get_hexstates", &Exporter::getHexstates, "Get a list of the HexState enum value names")
            .def("get_attribute_mapping", &Exporter::getAttributeMapping, "Get a attrname => (encname, offset, n, vmax)");
    }
}
