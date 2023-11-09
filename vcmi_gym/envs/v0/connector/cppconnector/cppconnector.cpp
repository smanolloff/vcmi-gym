#include <pybind11/functional.h>

#include "common.h"
#include "pyclient.h" // "vendor" header file

void _start(
    const P_RenderAnsiCBCB &p_renderansicbcb,
    const P_ResetCBCB &p_resetcbcb,
    const P_SysCBCB &p_syscbcb,
    const P_ActionCBCB &p_actioncbcb,
    const P_ResultCB &p_resultcb,
    const std::string mapname
) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    // Gym env code gives us callbacks
    // VCMI call gives us callbacks
    //
    // Neither cares about pybind11 and GIL:
    // this is is handled here.
    //
    // VCMI code calls and is called by raw, unwrapped callbacks
    // (see aitypes.h)
    //
    // Gym code calls and is called by wrapped callbacks
    // (see common.h)
    //
    LOG("create wrappers around callbacks")

    // Create PyCB from the given p_resultCB
    const MMAI::ResultCB resultcb = [&p_resultcb](const MMAI::Result &r) {
        LOG("[resultcb] start");

        LOG("[resultcb] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[resultcb] Convert r -> pgr");
        auto pgs = P_State(r.state.size());
        auto md = pgs.mutable_data();
        for (int i=0; i<r.state.size(); i++)
            md[i] = r.state[i].norm;

        auto pgr = P_Result(pgs,
            r.errmask,
            r.dmgDealt,
            r.dmgReceived,
            r.unitsLost,
            r.unitsKilled,
            r.valueLost,
            r.valueKilled,
            r.ended,
            r.victory
        );

        LOG("[resultcb] Call p_resultcb(pgr)");
        p_resultcb(pgr);
        LOG("[resultcb] return");
    };


    // Create PyCBInit from the given p_actioncbcb
    const MMAI::ActionCBCB actioncbcb = [&p_actioncbcb](MMAI::ActionCB &actioncb) {
        // Create WCppCB from the given CppCB
        P_ActionCB p_actioncb = [actioncb](P_Action pa) {
            //                         ^^^^^
            //                         NOT a reference!
            // The reference is to a local var within AISimulator::init
            // and becomes dangling as soon as AISimulator::init returns
            // (as soon as p_actioncbcb returns)
            LOG("[p_actioncb] start");

            // NOTE: we already have GIL, actioncb is called from python

            LOG("[p_actioncb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[p_actioncb] Convert pa -> a");
            auto a = static_cast<MMAI::Action>(pa);

            LOG("[p_actioncb] Call actioncb(a)");
            actioncb(a);
            LOG("[p_actioncb] return");
        };

        // NOTE: this may seem to work withoug GIL, but it is
        //       actually required (I once got an explicit error about it)
        LOG("[actioncbcb] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[actioncbcb] call p_actioncbcb(p_actioncb)");
        p_actioncbcb(p_actioncb);
        LOG("[actioncbcb] return");
    };


    // Create PyCBSysInit from the given p_syscbcb
    // see actioncbcb notes about GIL and lambda var scoping
    const MMAI::SysCBCB syscbcb = [&p_syscbcb](MMAI::SysCB &syscb) {
        // Create WCppSysCB from the given syscb
        P_SysCB p_syscb = [syscb](std::string action) {
            LOG("[p_syscb] start");

            LOG("[p_syscb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[p_syscb] call syscb");
            syscb(action);
            LOG("[p_syscb] return");
        };

        LOG("[syscbcb] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[syscbcb] call p_syscbcb(p_syscb)");
        p_syscbcb(p_syscb);

        // Release GIL here instead of before calling start_vcmi()
        // (PySysCBInit is invoked in the main thread!)
        LOG("[syscbcb] release Python GIL");
        py::gil_scoped_release release;
        LOG("[syscbcb] return");
    };

    // Create ResetCBCB from the given resetcb
    const MMAI::ResetCBCB resetcbcb = [&p_resetcbcb](MMAI::ResetCB &resetcb) {
        // Convert CppResetCBInit -> WCppResetCBInit
        P_ResetCB p_resetcb = [resetcb]() {
            LOG("[p_resetcb] start");

            LOG("[p_resetcb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[p_resetcb] call resetcb");
            resetcb();
            LOG("[p_resetcb] return");
        };

        LOG("[resetcbcb] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[resetcbcb] call p_resetcbcb(p_resetcb)");
        p_resetcbcb(p_resetcb);
        LOG("[resetcbcb] return");
    };

    // Create RenderAnsiCBCB from the given renderansicb
    const MMAI::RenderAnsiCBCB renderansicbcb = [&p_renderansicbcb](MMAI::RenderAnsiCB &renderansicb) {
        // Convert CppResetCBInit -> WCppResetCBInit
        P_RenderAnsiCB p_renderansicb = [renderansicb]() {
            LOG("[p_renderansicb] start");

            LOG("[p_renderansicb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[p_renderansicb] return renderansicb()");
            return renderansicb();
        };

        LOG("[resetcbcb] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[resetcbcb] call p_renderansicbcb(p_renderansicb)");
        p_renderansicbcb(p_renderansicb);
        LOG("[resetcbcb] return");
    };

    // NOTE: GIL is released when CPP calls the sysinit callback
    // LOG("release Python GIL");
    // py::gil_scoped_release release;

    auto cbprovider = MMAI::CBProvider{renderansicbcb, resetcbcb, syscbcb, actioncbcb, resultcb};

    // TODO: config values
    // std::string resdir = "/Users/simo/Projects/vcmi-/vcmi_/envs/v0/vcmi/build/bin";

    LOG("Start VCMI");
    start_vcmi(cbprovider, mapname);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}


void start(
    const P_RenderAnsiCBCB &p_renderansicbcb,
    const P_ResetCBCB &p_resetcbcb,
    const P_SysCBCB &p_syscbcb,
    const P_ActionCBCB &p_actioncbcb,
    const P_ResultCB &p_resultcb,
    const std::string mapname,
    const std::string loglevel
) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start (main thread)");

    // TODO: read config
    std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";

    // This must happen in the main thread (SDL requires it)
    LOG("call preinit_vcmi(...)");
    preinit_vcmi(resdir, loglevel);

    LOG("launch new thread");
    boost::thread t([p_renderansicbcb, p_resetcbcb, p_syscbcb, p_actioncbcb, p_resultcb, mapname]() {
        _start(p_renderansicbcb, p_resetcbcb, p_syscbcb, p_actioncbcb, p_resultcb, mapname);
    });

    LOG("return");
}

PYBIND11_MODULE(cppconnector, m) {
    m.def("start", &start, "Start VCMI");
    m.def("get_state_size", &get_state_size, "Get number of elements in state");
    m.def("get_action_max", &get_action_max, "Get max expected value of action");
    m.def("get_error_mapping", &get_error_mapping, "Get available error names and flags");

    py::class_<P_Result>(m, "PyGymResult")
        .def("state", &P_Result::state)
        .def("errmask", &P_Result::errmask)
        .def("dmg_dealt", &P_Result::dmg_dealt)
        .def("dmg_received", &P_Result::dmg_received)
        .def("units_lost", &P_Result::units_lost)
        .def("units_killed", &P_Result::units_killed)
        .def("value_lost", &P_Result::value_lost)
        .def("value_killed", &P_Result::value_killed)
        .def("is_battle_over", &P_Result::is_battle_over)
        .def("is_victorious", &P_Result::is_victorious);
}
