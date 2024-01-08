#include "connector.h"
#include "myclient.h"

Connector::Connector(
    const std::string mapname_,
    const std::string loglevelGlobal_,
    const std::string loglevelAI_,
    const std::string attacker_,
    const std::string defender_,
    const std::string attackerModel_,
    const std::string defenderModel_
) : mapname(mapname_),
    loglevelGlobal(loglevelGlobal_),
    loglevelAI(loglevelAI_),
    attacker(attacker_),
    defender(defender_),
    attackerModel(attackerModel_),
    defenderModel(defenderModel_),
    baggage(std::make_unique<MMAI::Export::Baggage>(initBaggage())) {}

MMAI::Export::Baggage Connector::initBaggage() {
    return MMAI::Export::Baggage([this](const MMAI::Export::Result* r) {
        return this->getAction(r);
    });
}

const P_Result Connector::convertResult(const MMAI::Export::Result* r) {
    LOG("Convert Result -> P_Result");

    auto ps = P_State(r->state.size());
    auto psmd = ps.mutable_data();
    for (int i=0; i < r->state.size(); i++)
        psmd[i] = r->state[i].norm;

    auto pam = P_ActMask(r->actmask.size());
    auto pammd = pam.mutable_data();
    for (int i=0; i < r->actmask.size(); i++)
        pammd[i] = r->actmask[i];

    return P_Result(
         r->type, ps, pam, r->errmask, r->side,
         r->dmgDealt, r->dmgReceived,
         r->unitsLost, r->unitsKilled,
         r->valueLost, r->valueKilled,
         r->side0ArmyValue, r->side1ArmyValue,
         r->ended, r->victory, r->ansiRender
    );
}

const P_Result Connector::reset() {
    assert(state == ConnectorState::AWAITING_ACTION);

    std::unique_lock lock(m);
    LOG("obtain lock: done");

    LOGSTR("set this->action = ", std::to_string(MMAI::Export::ACTION_RESET));
    action = MMAI::Export::ACTION_RESET;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("cond.notify_one()");
    cond.notify_one();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond.wait(lock)");
        cond.wait(lock);
        LOG("cond.wait(lock): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::REGULAR);

    LOG("release lock (return)");
    LOG("return P_Result");
    return convertResult(result);
}

const std::string Connector::renderAnsi() {
    assert(state == ConnectorState::AWAITING_ACTION);

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    LOGSTR("set this->action = ", std::to_string(MMAI::Export::ACTION_RENDER_ANSI));
    action = MMAI::Export::ACTION_RENDER_ANSI;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("cond.notify_one()");
    cond.notify_one();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond.wait(lock)");
        cond.wait(lock);
        LOG("cond.wait(lock): done");

        LOG("acquire Python GIL (scope-auto)");
    }

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::ANSI_RENDER);

    LOG("release lock (return)");
    LOG("return P_Result");
    return result->ansiRender;
}

const P_Result Connector::act(MMAI::Export::Action a) {
    assert(state == ConnectorState::AWAITING_ACTION);

    // Prevent control actions via `step`
    assert(a > 0);

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    LOGSTR("set this->action = ", std::to_string(a));
    action = a;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("cond.notify_one()");
    cond.notify_one();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond.wait(lock)");
        cond.wait(lock);
        LOG("cond.wait(lock): done");

        LOG("acquire Python GIL (scope-auto)");
    }

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::REGULAR);

    LOG("release lock (return)");
    LOG("return P_Result");
    return convertResult(result);
}

const P_Result Connector::start() {
    assert(state == ConnectorState::NEW);

    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    // TODO: read config
    //
    std::string gymdir = "/Users/simo/Projects/vcmi-gym";

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    // auto oldcwd = std::filesystem::current_path();

    // This must happen in the main thread (SDL requires it)
    LOG("call init_vcmi(...)");
    init_vcmi(
        baggage.get(),
        gymdir,
        mapname,
        loglevelGlobal,
        loglevelAI,
        attacker,
        defender,
        attackerModel,
        defenderModel,
        true  // VCMI GUI requires main thread (where SDL loop runs forever)
    );

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("launch new thread");
    vcmithread = std::thread([] {
        LOG("[thread] Start VCMI");
        start_vcmi();
        assert(false); // should never happen
    });

    // LOG("detach the newly created thread...")
    // vcmithread.detach();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond.wait(lock)");
        cond.wait(lock);
        LOG("cond.wait(lock): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    // auto newcwd = std::filesystem::current_path();
    // std::cout << "OLDCWD: " << oldcwd << "\nNEWCWD: " << newcwd << "\n";

    // NOTE: changing CWD here *sometimes* fails with exception:
    // std::__1::ios_base::failure: could not open file: unspecified iostream_category error
    // (sometimes = fails on benchmark, works on test...)
    //
    // LOGSTR("Change cwd back to", oldcwd.string());
    // std::filesystem::current_path(oldcwd);

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result->type == MMAI::Export::ResultType::REGULAR);

    LOG("release lock (return)");
    LOG("return P_Result");

    return convertResult(result);
}

MMAI::Export::Action Connector::getAction(const MMAI::Export::Result* r) {

    LOG("acquire Python GIL");
    py::gil_scoped_acquire acquire;

    LOG("obtain lock");
    std::unique_lock lock(m);
    LOG("obtain lock: done");

    assert(state == ConnectorState::AWAITING_RESULT);

    LOG("set this->result = r");
    result = r;

    LOG("set state = AWAITING_ACTION");
    state = ConnectorState::AWAITING_ACTION;

    LOG("cond.notify_one()");
    cond.notify_one();

    assert(state == ConnectorState::AWAITING_ACTION);

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        // Now wait again (will unblock once step/reset have been called)
        LOG("cond.wait(lock)");
        cond.wait(lock);
        LOG("cond.wait(lock): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    assert(state == ConnectorState::AWAITING_RESULT);

    LOG("release lock (return)");
    LOGSTR("return Action: ", std::to_string(action));
    return action;
}

PYBIND11_MODULE(connector, m) {
    py::class_<P_Result>(m, "P_Result")
        .def("get_state", &P_Result::get_state)
        .def("get_actmask", &P_Result::get_actmask)
        .def("get_errmask", &P_Result::get_errmask)
        .def("get_side", &P_Result::get_side)
        .def("get_dmg_dealt", &P_Result::get_dmg_dealt)
        .def("get_dmg_received", &P_Result::get_dmg_received)
        .def("get_units_lost", &P_Result::get_units_lost)
        .def("get_units_killed", &P_Result::get_units_killed)
        .def("get_value_lost", &P_Result::get_value_lost)
        .def("get_value_killed", &P_Result::get_value_killed)
        .def("get_side0_army_value", &P_Result::get_side0_army_value)
        .def("get_side1_army_value", &P_Result::get_side1_army_value)
        .def("get_is_battle_over", &P_Result::get_is_battle_over)
        .def("get_is_victorious", &P_Result::get_is_victorious);

    // py::class_<Child, std::shared_ptr<Child>>(m, "Child");

    py::class_<Connector, std::unique_ptr<Connector>>(m, "Connector")
        .def(py::init<
            const std::string &, // mapname
            const std::string &, // loglevelGlobal
            const std::string &, // loglevelAI
            const std::string &, // attacker
            const std::string &, // defender
            const std::string &, // attackerModel
            const std::string &  // defenderModel
        >())
        .def("start", &Connector::start)
        .def("reset", &Connector::reset)
        .def("act", &Connector::act)
        .def("renderAnsi", &Connector::renderAnsi);
}
