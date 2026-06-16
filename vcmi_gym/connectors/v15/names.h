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
    std::string Attr(S15::Graph::EdgeAttributes::Global_To_Player a);
    std::string Attr(S15::Graph::EdgeAttributes::Player_To_Global a);
    std::string Attr(S15::Graph::EdgeAttributes::Global_To_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_To_Global a);
    std::string Attr(S15::Graph::EdgeAttributes::Global_To_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Hex_To_Global a);
    std::string Attr(S15::Graph::EdgeAttributes::Global_To_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Player_Owns_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_OwnedBy_Player a);
    std::string Attr(S15::Graph::EdgeAttributes::Hex_Adjacent_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_ActsBefore_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_MeleeDmg_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_ShootDmg_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_Blocks_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_Occupies_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Hex_OccupiedBy_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_By_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_Has_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_EndsAt_Hex a);
    std::string Attr(S15::Graph::EdgeAttributes::Hex_IsEndOf_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Action_Blocks_Unit a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_BlockedBy_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_BecomesMeleeThreatAfter_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_BecomesShootThreatAfter_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_IsMeleedBy_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_IsShotBy_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_BecomesMeleeTargetAfter_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Unit_BecomesShootTargetAfter_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Hex_BecomesMeleeTargetAfter_Action a);
    std::string Attr(S15::Graph::EdgeAttributes::Hex_BecomesShootTargetAfter_Action a);
    static_assert(static_cast<int>(S15::Graph::ElementType::_count) == 35);
}
