#pragma once

#include "schema/v15/graph.h"
#include "schema/v15/types.h"

namespace Connector::V15::Names {
    namespace S15 = MMAI::Schema::V15;

    std::string CombatResult(S15::CombatResult cr);
    std::string ActionType(S15::ActionType at);

    std::string Encoding(S15::Encoding e);

    std::string Attr(S15::Graph::NodeAttributes::Global a);
    std::string Attr(S15::Graph::NodeAttributes::Player a);
    std::string Attr(S15::Graph::NodeAttributes::Unit a);
    std::string Attr(S15::Graph::NodeAttributes::Hex a);
    std::string Attr(S15::Graph::NodeAttributes::Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Global_Has_Player a);
    std::string Attr(S15::Graph::EdgeAttributes::Global_Has_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Global_Has_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Player_Owns_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Hex_Adjacent_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_ActsBefore_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_MeleeDmg_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_ShootDmg_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_Blocks_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_Occupies_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_By_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_EndsAt_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_Blocks_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_ExposesToMeleeFrom_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_ExposesToShootFrom_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_Melees_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_Shoots_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_EnablesMeleeAt_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_EnablesShootAt_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_EnablesMeleeAt_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_EnablesShootAt_Hex a);
}
