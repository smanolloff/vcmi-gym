#include "names.h"
#include "schema/v15/types.h"

namespace Connector::V15::Names {
    namespace NA = S15::Graph::NodeAttributes;
    namespace EA = S15::Graph::EdgeAttributes;

    std::string CombatResult(S15::CombatResult cr) {
        switch(cr) {
        case S15::CombatResult::LEFT_WINS: return "LEFT_WINS";
        case S15::CombatResult::RIGHT_WINS: return "RIGHT_WINS";
        case S15::CombatResult::DRAW: return "DRAW";
        case S15::CombatResult::NONE: return "NONE";
        default:
            throw std::runtime_error("Names: unexpected CombatResult: " + std::to_string(EI(cr)));
        }
    }

    std::string ActionType(S15::ActionType at) {
        switch (at) {
        case S15::ActionType::RETREAT: return "RETREAT";
        case S15::ActionType::WAIT: return "WAIT";
        case S15::ActionType::DEFEND: return "DEFEND";
        case S15::ActionType::MOVE: return "MOVE";
        case S15::ActionType::AMOVE: return "AMOVE";
        case S15::ActionType::SHOOT: return "SHOOT";
        default:
            throw std::runtime_error("Names: unexpected ActionType: " + std::to_string(EI(at)));
        }
    }

    std::string Encoding(S15::Encoding e) {
        switch(e) {
        case S15::Encoding::CATEGORICAL: return "CATEGORICAL";
        case S15::Encoding::LINNORM: return "LINNORM";
        case S15::Encoding::RAW: return "RAW";
        default:
            throw std::runtime_error("Names: unexpected Encoding: " + std::to_string(EI(e)));
        }
    }

    std::string Attr(NA::Global a) {
        switch(a) {
        case NA::Global::BATTLE_WINNER: return "BATTLE_WINNER";
        case NA::Global::BATTLE_ROUND: return "BATTLE_ROUND";
        case NA::Global::HAS_UPPER_TOWER: return "HAS_UPPER_TOWER";
        case NA::Global::HAS_MIDDLE_TOWER: return "HAS_MIDDLE_TOWER";
        case NA::Global::HAS_BOTTOM_TOWER: return "HAS_BOTTOM_TOWER";
        case NA::Global::HAS_GATE_CORPSE: return "HAS_GATE_CORPSE";
        case NA::Global::HAS_BRIDGE_CORPSE: return "HAS_BRIDGE_CORPSE";
        default:
            throw std::runtime_error("Names: unexpected Global attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(NA::Player a) {
        switch(a) {
        case NA::Player::BATTLE_SIDE: return "BATTLE_SIDE";
        case NA::Player::IS_ACTIVE: return "IS_ACTIVE";
        case NA::Player::ARMY_VALUE_NOW_REL0: return "ARMY_VALUE_NOW_REL0";
        case NA::Player::ARMY_VALUE_NOW_REL: return "ARMY_VALUE_NOW_REL";
        case NA::Player::ARMY_HP_NOW_REL: return "ARMY_HP_NOW_REL";
        case NA::Player::VALUE_KILLED_NOW_REL: return "VALUE_KILLED_NOW_REL";
        case NA::Player::VALUE_LOST_NOW_REL: return "VALUE_LOST_NOW_REL";
        case NA::Player::DMG_DEALT_NOW_REL: return "DMG_DEALT_NOW_REL";
        case NA::Player::DMG_RECEIVED_NOW_REL: return "DMG_RECEIVED_NOW_REL";
        default:
            throw std::runtime_error("Names: unexpected Player attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(NA::Unit a) {
        switch(a) {
        case NA::Unit::VALUE_REL: return "VALUE_REL";
        case NA::Unit::SHOTS: return "SHOTS";
        case NA::Unit::DMG_UNCERTAINTY: return "DMG_UNCERTAINTY";
        case NA::Unit::IS_ACTIVE: return "IS_ACTIVE";
        case NA::Unit::IS_ENEMY: return "IS_ENEMY";
        case NA::Unit::IS_SLEEPING: return "IS_SLEEPING";
        case NA::Unit::IS_WAR_MACHINE: return "IS_WAR_MACHINE";
        case NA::Unit::HAS_ADDITIONAL_ATTACK: return "HAS_ADDITIONAL_ATTACK";
        case NA::Unit::HAS_ALL_AROUND_ATTACK: return "HAS_ALL_AROUND_ATTACK";
        case NA::Unit::HAS_BLOCKS_RETALIATION: return "HAS_BLOCKS_RETALIATION";
        case NA::Unit::HAS_DEATH_CLOUD: return "HAS_DEATH_CLOUD";
        case NA::Unit::HAS_DOUBLE_DAMAGE_CHANCE: return "HAS_DOUBLE_DAMAGE_CHANCE";
        case NA::Unit::HAS_FIREBALL: return "HAS_FIREBALL";
        case NA::Unit::HAS_FLYING: return "HAS_FLYING";
        case NA::Unit::HAS_LIFE_DRAIN: return "HAS_LIFE_DRAIN";
        case NA::Unit::HAS_NON_LIVING: return "HAS_NON_LIVING";
        case NA::Unit::HAS_NO_MELEE_PENALTY: return "HAS_NO_MELEE_PENALTY";
        case NA::Unit::HAS_RETURN_AFTER_STRIKE: return "HAS_RETURN_AFTER_STRIKE";
        case NA::Unit::HAS_THREE_HEADED_ATTACK: return "HAS_THREE_HEADED_ATTACK";
        case NA::Unit::HAS_TWO_HEX_ATTACK_BREATH: return "HAS_TWO_HEX_ATTACK_BREATH";
        case NA::Unit::HAS_AGE: return "HAS_AGE";
        case NA::Unit::HAS_AGE_ATTACK: return "HAS_AGE_ATTACK";
        case NA::Unit::HAS_BIND: return "HAS_BIND";
        case NA::Unit::HAS_BIND_ATTACK: return "HAS_BIND_ATTACK";
        case NA::Unit::HAS_BLIND: return "HAS_BLIND";
        case NA::Unit::HAS_BLIND_ATTACK: return "HAS_BLIND_ATTACK";
        case NA::Unit::HAS_CURSE: return "HAS_CURSE";
        case NA::Unit::HAS_CURSE_ATTACK: return "HAS_CURSE_ATTACK";
        case NA::Unit::HAS_DISPEL_ATTACK: return "HAS_DISPEL_ATTACK";
        case NA::Unit::HAS_PETRIFY: return "HAS_PETRIFY";
        case NA::Unit::HAS_PETRIFY_ATTACK: return "HAS_PETRIFY_ATTACK";
        case NA::Unit::HAS_POISON: return "HAS_POISON";
        case NA::Unit::HAS_POISON_ATTACK: return "HAS_POISON_ATTACK";
        case NA::Unit::HAS_WEAKNESS: return "HAS_WEAKNESS";
        case NA::Unit::HAS_WEAKNESS_ATTACK: return "HAS_WEAKNESS_ATTACK";
        default:
            throw std::runtime_error("Names: unexpected Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(NA::Hex a) {
        switch(a) {
        case NA::Hex::Y_COORD: return "Y_COORD";
        case NA::Hex::X_COORD: return "X_COORD";
        case NA::Hex::IS_PASSABLE: return "IS_PASSABLE";
        case NA::Hex::IS_STOPPING: return "IS_STOPPING";
        case NA::Hex::IS_DAMAGING_L: return "IS_DAMAGING_L";
        case NA::Hex::IS_DAMAGING_R: return "IS_DAMAGING_R";
        case NA::Hex::IS_SIEGE_GATE: return "IS_SIEGE_GATE";
        case NA::Hex::IS_SIEGE_BRIDGE: return "IS_SIEGE_BRIDGE";
        case NA::Hex::IS_OBSTACLE: return "IS_OBSTACLE";
        case NA::Hex::WALL_HEALTH: return "WALL_HEALTH";
        default:
            throw std::runtime_error("Names: unexpected Hex attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(NA::Action a) {
        switch(a) {
        case NA::Action::ACTION_TYPE: return "ACTION_TYPE";
        case NA::Action::IS_ACTIVE: return "IS_ACTIVE";
        default:
            throw std::runtime_error("Names: unexpected Action attribute: " + std::to_string(EI(a)));
        }
    }


    std::string Attr(EA::Global_To_Player a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Global_To_Player attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Player_To_Global a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Player_To_Global attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Global_To_Unit a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Global_To_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_To_Global a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Unit_To_Global attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Global_To_Hex a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Global_To_Hex attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Hex_To_Global a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Hex_To_Global attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Global_To_Action a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Global_To_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Player_Owns_Unit a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Player_Owns_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_OwnedBy_Player a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Unit_OwnedBy_Player attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Hex_Adjacent_Hex a) {
        switch(a) {
        case EA::Hex_Adjacent_Hex::DIRECTION: return "DIRECTION";
        default:
            throw std::runtime_error("Names: unexpected Hex_Adjacent_Hex attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_ActsBefore_Unit a) {
        switch(a) {
        case EA::Unit_ActsBefore_Unit::TIMES: return "TIMES";
        default:
            throw std::runtime_error("Names: unexpected Unit_ActsBefore_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_MeleeDmg_Unit a) {
        switch(a) {
        case EA::Unit_MeleeDmg_Unit::ESTIMATED_ATTACKER_HPDIFF_REL_SELF: return "ESTIMATED_ATTACKER_HPDIFF_REL_SELF";
        case EA::Unit_MeleeDmg_Unit::ESTIMATED_ATTACKER_HPDIFF_REL_BF: return "ESTIMATED_ATTACKER_HPDIFF_REL_BF";
        case EA::Unit_MeleeDmg_Unit::ESTIMATED_DEFENDER_HPDIFF_REL_SELF: return "ESTIMATED_DEFENDER_HPDIFF_REL_SELF";
        case EA::Unit_MeleeDmg_Unit::ESTIMATED_DEFENDER_HPDIFF_REL_BF: return "ESTIMATED_DEFENDER_HPDIFF_REL_BF";
        case EA::Unit_MeleeDmg_Unit::ESTIMATED_NET_VALUE_REL_BF: return "ESTIMATED_NET_VALUE_REL_BF";
        default:
            throw std::runtime_error("Names: unexpected Unit_MeleeDmg_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_ShootDmg_Unit a) {
        switch(a) {
        case EA::Unit_ShootDmg_Unit::ESTIMATED_ATTACKER_HPDIFF_REL_SELF: return "ESTIMATED_ATTACKER_HPDIFF_REL_SELF";
        case EA::Unit_ShootDmg_Unit::ESTIMATED_ATTACKER_HPDIFF_REL_BF: return "ESTIMATED_ATTACKER_HPDIFF_REL_BF";
        case EA::Unit_ShootDmg_Unit::ESTIMATED_DEFENDER_HPDIFF_REL_SELF: return "ESTIMATED_DEFENDER_HPDIFF_REL_SELF";
        case EA::Unit_ShootDmg_Unit::ESTIMATED_DEFENDER_HPDIFF_REL_BF: return "ESTIMATED_DEFENDER_HPDIFF_REL_BF";
        case EA::Unit_ShootDmg_Unit::ESTIMATED_NET_VALUE_REL_BF: return "ESTIMATED_NET_VALUE_REL_BF";
        default:
            throw std::runtime_error("Names: unexpected Unit_ShootDmg_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_Blocks_Unit a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Unit_Blocks_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_Occupies_Hex a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Unit_Occupies_Hex attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Hex_OccupiedBy_Unit a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Hex_OccupiedBy_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Action_By_Unit a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Action_By_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_Has_Action a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Unit_Has_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Action_EndsAt_Hex a) {
        switch(a) {
        case EA::Action_EndsAt_Hex::IS_REAR: return "IS_REAR";
        default:
            throw std::runtime_error("Names: unexpected Action_EndsAt_Hex attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Hex_IsEndOf_Action a) {
        switch(a) {
        case EA::Hex_IsEndOf_Action::IS_REAR: return "IS_REAR";
        default:
            throw std::runtime_error("Names: unexpected Hex_IsEndOf_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Action_Blocks_Unit a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Action_Blocks_Unit attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_BecomesMeleeThreatAfter_Action a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Unit_BecomesMeleeThreatAfter_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_BecomesShootThreatAfter_Action a) {
        switch(a) {
        case EA::Unit_BecomesShootThreatAfter_Action::DMG_MULT: return "DMG_MULT";
        default:
            throw std::runtime_error("Names: unexpected Unit_BecomesShootThreatAfter_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_IsMeleedBy_Action a) {
        switch(a) {
        case EA::Unit_IsMeleedBy_Action::IS_PRIMARY_TARGET: return "IS_PRIMARY_TARGET";
        default:
            throw std::runtime_error("Names: unexpected Unit_IsMeleedBy_Action attribute: " + std::to_string(EI(a)));
        }
     }

    std::string Attr(EA::Unit_IsShotBy_Action a) {
        switch(a) {
        case EA::Unit_IsShotBy_Action::IS_PRIMARY_TARGET: return "IS_PRIMARY_TARGET";
        default:
            throw std::runtime_error("Names: unexpected Unit_IsShotBy_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_BecomesMeleeTargetAfter_Action a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Unit_BecomesMeleeTargetAfter_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Unit_BecomesShootTargetAfter_Action a) {
        switch(a) {
        case EA::Unit_BecomesShootTargetAfter_Action::DMG_MULT: return "DMG_MULT";
        default:
            throw std::runtime_error("Names: unexpected Unit_BecomesShootTargetAfter_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Hex_BecomesMeleeTargetAfter_Action a) {
        switch(a) {
        default:
            throw std::runtime_error("Names: unexpected Hex_BecomesMeleeTargetAfter_Action attribute: " + std::to_string(EI(a)));
        }
    }

    std::string Attr(EA::Hex_BecomesShootTargetAfter_Action a) {
        switch(a) {
        case EA::Hex_BecomesShootTargetAfter_Action::DMG_MULT: return "DMG_MULT";
        default:
            throw std::runtime_error("Names: unexpected Hex_BecomesShootTargetAfter_Action attribute: " + std::to_string(EI(a)));
        }
    }


}
