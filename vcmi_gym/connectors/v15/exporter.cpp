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

// Linux builds fail without this include (used in types.h)
#include "exporter.h"

#include "names.h"
#include "schema/v15/constants.h"
#include "schema/v15/graph.h"
#include "schema/v15/types.h"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <unordered_map>

namespace Connector::V15 {
    namespace S15 = MMAI::Schema::V15;

    namespace
    {
        using AttrEncoding = std::tuple<std::string, std::string, int, int>;

        using ET = S15::Graph::ElementType;
        namespace NA = S15::Graph::NodeAttributes;
        namespace EA = S15::Graph::EdgeAttributes;


        std::vector<AttrEncoding> GetEncoding(S15::Graph::ElementType type)
        {
            auto process = [](const auto & encoding)
            {
                auto res = std::vector<AttrEncoding>{};
                for (const auto & [a, e, n, vmax] : encoding)
                    res.push_back({Names::Attr(a), Names::Encoding(e), n, vmax});
                return res;
            };

            switch(type)
            {
            case ET::NODE_GLOBAL: return process(S15::EncodingTraits<NA::Global>::encoding);
            case ET::NODE_PLAYER: return process(S15::EncodingTraits<NA::Player>::encoding);
            case ET::NODE_UNIT: return process(S15::EncodingTraits<NA::Unit>::encoding);
            case ET::NODE_HEX: return process(S15::EncodingTraits<NA::Hex>::encoding);
            case ET::NODE_ACTION: return process(S15::EncodingTraits<NA::Action>::encoding);
            case ET::EDGE_GLOBAL_TO_PLAYER: return process(S15::EncodingTraits<EA::Global_To_Player>::encoding);
            case ET::EDGE_PLAYER_TO_GLOBAL: return process(S15::EncodingTraits<EA::Player_To_Global>::encoding);
            case ET::EDGE_GLOBAL_TO_UNIT: return process(S15::EncodingTraits<EA::Global_To_Unit>::encoding);
            case ET::EDGE_UNIT_TO_GLOBAL: return process(S15::EncodingTraits<EA::Unit_To_Global>::encoding);
            case ET::EDGE_GLOBAL_TO_HEX: return process(S15::EncodingTraits<EA::Global_To_Hex>::encoding);
            case ET::EDGE_HEX_TO_GLOBAL: return process(S15::EncodingTraits<EA::Hex_To_Global>::encoding);
            case ET::EDGE_GLOBAL_TO_ACTION: return process(S15::EncodingTraits<EA::Global_To_Action>::encoding);
            case ET::EDGE_PLAYER_OWNS_UNIT: return process(S15::EncodingTraits<EA::Player_Owns_Unit>::encoding);
            case ET::EDGE_UNIT_OWNED_BY_PLAYER: return process(S15::EncodingTraits<EA::Unit_OwnedBy_Player>::encoding);
            case ET::EDGE_HEX_ADJACENT_HEX: return process(S15::EncodingTraits<EA::Hex_Adjacent_Hex>::encoding);
            case ET::EDGE_UNIT_ACTS_BEFORE_UNIT: return process(S15::EncodingTraits<EA::Unit_ActsBefore_Unit>::encoding);
            case ET::EDGE_UNIT_MELEE_DMG_UNIT: return process(S15::EncodingTraits<EA::Unit_MeleeDmg_Unit>::encoding);
            case ET::EDGE_UNIT_SHOOT_DMG_UNIT: return process(S15::EncodingTraits<EA::Unit_ShootDmg_Unit>::encoding);
            case ET::EDGE_UNIT_BLOCKS_UNIT: return process(S15::EncodingTraits<EA::Unit_Blocks_Unit>::encoding);
            case ET::EDGE_UNIT_OCCUPIES_HEX: return process(S15::EncodingTraits<EA::Unit_Occupies_Hex>::encoding);
            case ET::EDGE_HEX_OCCUPIED_BY_UNIT: return process(S15::EncodingTraits<EA::Hex_OccupiedBy_Unit>::encoding);
            case ET::EDGE_ACTION_BY_UNIT: return process(S15::EncodingTraits<EA::Action_By_Unit>::encoding);
            case ET::EDGE_UNIT_HAS_ACTION: return process(S15::EncodingTraits<EA::Unit_Has_Action>::encoding);
            case ET::EDGE_ACTION_ENDS_AT_HEX: return process(S15::EncodingTraits<EA::Action_EndsAt_Hex>::encoding);
            case ET::EDGE_HEX_IS_END_OF_ACTION: return process(S15::EncodingTraits<EA::Hex_IsEndOf_Action>::encoding);
            case ET::EDGE_ACTION_BLOCKS_UNIT: return process(S15::EncodingTraits<EA::Action_Blocks_Unit>::encoding);
            case ET::EDGE_UNIT_BECOMES_MELEE_THREAT_AFTER_ACTION: return process(S15::EncodingTraits<EA::Unit_BecomesMeleeThreatAfter_Action>::encoding);
            case ET::EDGE_UNIT_BECOMES_SHOOT_THREAT_AFTER_ACTION: return process(S15::EncodingTraits<EA::Unit_BecomesShootThreatAfter_Action>::encoding);
            case ET::EDGE_UNIT_IS_MELEED_BY_ACTION: return process(S15::EncodingTraits<EA::Unit_IsMeleedBy_Action>::encoding);
            case ET::EDGE_UNIT_IS_SHOT_BY_ACTION: return process(S15::EncodingTraits<EA::Unit_IsShotBy_Action>::encoding);
            case ET::EDGE_UNIT_BECOMES_MELEE_TARGET_AFTER_ACTION: return process(S15::EncodingTraits<EA::Unit_BecomesMeleeTargetAfter_Action>::encoding);
            case ET::EDGE_UNIT_BECOMES_SHOOT_TARGET_AFTER_ACTION: return process(S15::EncodingTraits<EA::Unit_BecomesShootTargetAfter_Action>::encoding);
            case ET::EDGE_HEX_BECOMES_MELEE_TARGET_AFTER_ACTION: return process(S15::EncodingTraits<EA::Hex_BecomesMeleeTargetAfter_Action>::encoding);
            case ET::EDGE_HEX_BECOMES_SHOOT_TARGET_AFTER_ACTION: return process(S15::EncodingTraits<EA::Hex_BecomesShootTargetAfter_Action>::encoding);
            default:
                throw std::runtime_error("GetEncoding: unexpected ElementType: " + std::to_string(EI(type)));
            }
        }
    }

    int getVersion() { return 15; }
    int getMaxRounds() { return static_cast<int>(S15::MAX_ROUNDS); }

    py::dict getActionTypes() {
        auto res = py::dict();
        for (int i=0; i < static_cast<int>(S15::ActionType::_count); i++)
            res[py::str(Names::ActionType(S15::ActionType(i)))] = i;
        return res;
    }

    py::dict getCombatResults() {
        auto res = py::dict();
        for (int i=0; i < static_cast<int>(S15::CombatResult::_count); i++)
            res[py::str(Names::CombatResult(S15::CombatResult(i)))] = i;
        return res;
    }

    py::dict getNodeTypes() {
        auto res = py::dict();
        for (const auto & [type, name, size] : S15::NODE_TYPES)
        {
            auto nodedict = py::dict();
            nodedict[py::str("size")] = size;

            auto attrs = py::list();
            for (const auto & [a, e, n, vmax] : GetEncoding(type))
            {
                attrs.append(py::dict(
                    py::arg("name") = py::str(a),
                    py::arg("encoding") = py::str(e),
                    py::arg("size") = n,
                    py::arg("vmax") = vmax
                ));
            }

            nodedict[py::str("attributes")] = attrs;
            res[py::str(name)] = nodedict;

        }
        return res;
    }

    py::dict getEdgeTypes() {
        auto nodetypes = std::unordered_map<S15::Graph::ElementType, std::string>{};

        for (const auto & [type, name, size] : S15::NODE_TYPES)
            nodetypes.emplace(type, name);

        auto res = py::dict();
        for (const auto & [type, name, endpoints, size] : S15::EDGE_TYPES)
        {
            const auto & [srcType, dstType] = endpoints;
            const auto & srcName = nodetypes.at(srcType);
            const auto & dstName = nodetypes.at(dstType);

            auto attrs = py::list();
            for (const auto & [a, e, n, vmax] : GetEncoding(type))
            {
                attrs.append(py::dict(
                    py::arg("name") = py::str(a),
                    py::arg("encoding") = py::str(e),
                    py::arg("size") = n,
                    py::arg("vmax") = vmax
                ));
            }

            auto edgedict = py::dict(
                py::arg("size")=size,
                py::arg("attributes")=attrs
            );

            const auto key = py::make_tuple(
                py::str(srcName),
                py::str(name),
                py::str(dstName)
            );
            res[key] = edgedict;

        }
        return res;
    }

    void bindExporterV15(py::module_ & m)
    {
        m.def("get_version", &getVersion);
        m.def("get_max_rounds", &getMaxRounds);
        m.def("get_action_types", &getActionTypes);
        m.def("get_node_types", &getNodeTypes);
        m.def("get_edge_types", &getEdgeTypes);
        m.def("get_combat_results", &getCombatResults);
    }

} // namespace Connector::V15

PYBIND11_MODULE(exporter_v15, m)
{
    Connector::V15::bindExporterV15(m);
}
