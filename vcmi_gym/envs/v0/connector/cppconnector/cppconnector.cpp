#include <pybind11/pybind11.h>
#include "cppconnector.h"
#include "aitypes.h"
#include "pyclient.h" // "vendor" header file

// const auto cc = std::make_unique<CppConnector>("pikemen.vmap", "trace");

std::unique_ptr<CppConnector> create_cppconnector(const std::string mapname_, const std::string loglevel_) {
    // cc = std::make_unique<CppConnector>(mapname_, loglevel_);
    return std::make_unique<CppConnector>(mapname_, loglevel_);
}

CppConnector::~CppConnector() {
    // not the issue - it seems it's not called
    LOG("---------------------------- DESTRUCTED ---------------------");
}

CppConnector::CppConnector(const std::string mapname_, const std::string loglevel_)
: mapname(mapname_), loglevel(loglevel_) {
    auto getaction = [this](MMAI::Result r) {
        LOG("GETACTIONNNNNNNNNNNN");
        return this->getAction(r);
        // return this->getActionDummy(r);
    };
    cbprovider->f_getAction = getaction;
    cbprovider->debugstr = "reassigned debugstr in CppConnector constructor";
}

// const MMAI::Action CppConnector::getActionDummy(MMAI::Result r) {
//     LOG("1");
//     result = r;
//     LOG("2");
//     state = ConnectorState::AWAITING_ACTION;
//     LOG("3");
//     LOGSTR("i=", std::to_string(i));
//     if (i++ > 5) {
//         cond.notify_one();
//     }
//     LOG("4");
//     auto a = MMAI::Action(3);
//     LOG("5");
//     return a;
// }

const P_Result CppConnector::convertResult(MMAI::Result r) {
    LOG("Convert Result -> P_Result");

    auto ps = P_State(r.state.size());
    auto md = ps.mutable_data();
    for (int i=0; i<r.state.size(); i++)
        md[i] = r.state[i].norm;

    return P_Result(
        result.type, ps, result.errmask, result.dmgDealt, result.dmgReceived,
        result.unitsLost, result.unitsKilled, result.valueLost,
        result.valueKilled, result.ended, result.victory, result.ansiRender
    );
}

const P_Result CppConnector::reset() {
    assert(state == ConnectorState::AWAITING_ACTION);

    // LOG("release Python GIL");
    // py::gil_scoped_release release;

    LOG("obtain lock1");
    std::unique_lock lock1(m1);
    LOG("obtain lock1: done");

    LOGSTR("set this->action = ", std::to_string(MMAI::ACTION_RESET));
    action = MMAI::ACTION_RESET;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("cond2.notify_one()");
    cond2.notify_one();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond1.wait(lock1)");
        cond1.wait(lock1);
        LOG("cond1.wait(lock1): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    LOG("obtain lock2");
    std::unique_lock lock2(m2);
    LOG("obtain lock2: done");

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result.type == MMAI::ResultType::REGULAR);

    LOG("release lock1 (return)");
    LOG("release lock2 (return)");
    LOG("return P_Result");
    return convertResult(result);
}

const std::string CppConnector::renderAnsi() {
    assert(state == ConnectorState::AWAITING_ACTION);

    LOG("obtain lock1");
    std::unique_lock lock1(m1);
    LOG("obtain lock1: done");

    LOGSTR("set this->action = ", std::to_string(MMAI::ACTION_RENDER_ANSI));
    action = MMAI::ACTION_RENDER_ANSI;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("cond2.notify_one()");
    cond2.notify_one();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond1.wait(lock1)");
        cond1.wait(lock1);
        LOG("cond1.wait(lock1): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    LOG("obtain lock2");
    std::unique_lock lock2(m2);
    LOG("obtain lock2: done");

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result.type == MMAI::ResultType::ANSI_RENDER);

    LOG("release lock1 (return)");
    LOG("release lock2 (return)");
    LOG("return P_Result");
    return result.ansiRender;
}

const P_Result CppConnector::act(MMAI::Action a) {
    assert(state == ConnectorState::AWAITING_ACTION);

    // Prevent control actions via `step`
    assert(a > 0);

    LOG("obtain lock1");
    std::unique_lock lock1(m1);
    LOG("obtain lock1: done");

    LOGSTR("set this->action = ", std::to_string(a));
    action = a;

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("cond2.notify_one()");
    cond2.notify_one();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond1.wait(lock1)");
        cond1.wait(lock1);
        LOG("cond1.wait(lock1): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    LOG("obtain lock2");
    std::unique_lock lock2(m2);
    LOG("obtain lock2: done");

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result.type == MMAI::ResultType::REGULAR);

    LOG("release lock1 (return)");
    LOG("release lock2 (return)");
    LOG("return P_Result");
    return convertResult(result);
}

const P_Result CppConnector::start() {
    assert(state == ConnectorState::NEW);

    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    // TODO: read config
    std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";

    LOG("obtain lock1");
    std::unique_lock lock1(m1);
    LOG("obtain lock1: done");

    // This must happen in the main thread (SDL requires it)
    LOG("call init_vcmi(...)");
    f_sys = init_vcmi(loglevel, cbprovider.get());

    LOG("set state = AWAITING_RESULT");
    state = ConnectorState::AWAITING_RESULT;

    LOG("launch new thread");
    auto map = mapname;
    vcmithread = std::thread([map]() {
        LOG("[thread] Start VCMI");
        start_vcmi(map);
        assert(false); // should never happen
    });

    // LOG("detach the newly created thread...")
    // vcmithread.detach();

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("cond1.wait(lock1)");
        cond1.wait(lock1);
        LOG("cond1.wait(lock1): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    assert(state == ConnectorState::AWAITING_ACTION);
    assert(result.type == MMAI::ResultType::REGULAR);

    LOG("release lock1 (return)");
    LOG("release lock2 (return)");
    LOG("return P_Result");

    // TODO: smart ref here
    return convertResult(result);
}

MMAI::Action CppConnector::getAction(MMAI::Result r) {

    LOG("acquire Python GIL");
    py::gil_scoped_acquire acquire;

    LOG("obtain lock1");
    std::unique_lock lock1(m1);
    LOG("obtain lock1: done");

    assert(state == ConnectorState::AWAITING_RESULT);

    LOG("set this->result = r");
    result = r;

    LOG("set state = AWAITING_ACTION");
    state = ConnectorState::AWAITING_ACTION;

    LOG("cond1.notify_one()");
    cond1.notify_one();

    LOG("obtain lock2");
    std::unique_lock lock2(m2);
    LOG("obtain lock2: done");

    LOG("release lock1");
    lock1.unlock();

    assert(state == ConnectorState::AWAITING_ACTION);

    {
        LOG("release Python GIL");
        py::gil_scoped_release release;

        // Now wait again (will unblock once step/reset have been called)
        LOG("cond2.wait(lock2)");
        cond2.wait(lock2);
        LOG("cond2.wait(lock2): done");

        LOG("acquire Python GIL (scope-auto)");
        // py::gil_scoped_acquire acquire2;
    }

    assert(state == ConnectorState::AWAITING_RESULT);

    LOG("release lock2 (return)");
    LOGSTR("return Action: ", std::to_string(action));
    return action;
}

static const int get_state_size() { return MMAI::STATE_SIZE; }
static const int get_n_actions() { return MMAI::N_ACTIONS; }
static const std::map<MMAI::ErrMask, std::tuple<std::string, std::string>> get_error_mapping() {
  std::map<MMAI::ErrMask, std::tuple<std::string, std::string>> res;

  for (auto& [_err, tuple] : MMAI::ERRORS) {
    res[std::get<0>(tuple)] = {std::get<1>(tuple), std::get<2>(tuple)};
  }

  return res;
}

PYBIND11_MODULE(cppconnector, m) {
    m.def("get_state_size", &get_state_size, "Get number of elements in state");
    m.def("get_n_actions", &get_n_actions, "Get max expected value of action");
    m.def("get_error_mapping", &get_error_mapping, "Get available error names and flags");
    m.def("create_cppconnector", &create_cppconnector, "");

    py::class_<P_Result>(m, "P_Result")
        .def("get_state", &P_Result::get_state)
        .def("get_errmask", &P_Result::get_errmask)
        .def("get_dmg_dealt", &P_Result::get_dmg_dealt)
        .def("get_dmg_received", &P_Result::get_dmg_received)
        .def("get_units_lost", &P_Result::get_units_lost)
        .def("get_units_killed", &P_Result::get_units_killed)
        .def("get_value_lost", &P_Result::get_value_lost)
        .def("get_value_killed", &P_Result::get_value_killed)
        .def("get_is_battle_over", &P_Result::get_is_battle_over)
        .def("get_is_victorious", &P_Result::get_is_victorious);

    py::class_<CppConnector>(m, "CppConnector")
        .def(py::init<const std::string &, const std::string &>())
        .def("start", &CppConnector::start)
        .def("reset", &CppConnector::reset)
        .def("act", &CppConnector::act)
        .def("renderAnsi", &CppConnector::renderAnsi);
}
