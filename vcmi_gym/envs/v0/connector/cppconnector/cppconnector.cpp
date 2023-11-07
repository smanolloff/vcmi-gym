#include <pybind11/functional.h>

#include "common.h"
#include "pyclient.h" // "vendor" header file

void _start(
    const WPyCBResetInit &wpycbresetinit,
    const WPyCBSysInit &wpycbsysinit,
    const WPyCBInit &wpycbinit,
    const WPyCB &wpycb,
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
    // TODO: this terminology is very confusing
    //       replace "W" with "P_" prefix
    //       (all python-interacting functions are P_, others are the raw)
    //
    // TODO: rename functions
    //       PyCB -> GymReportCB
    //       CppCB -> VcmiActCB
    //       PyCBInit -> GymInitCB
    //       ...

    LOG("create wrappers around callbacks")

    // Create PyCB from the given WPyCB
    const MMAI::PyCB pycb = [&wpycb](const MMAI::GymResult &gr) {
        LOG("[pycb] start");

        LOG("[pycb] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[pycb] Convert gr -> pgr");
        auto pgs = PyGymState(gr.state.size());
        auto md = pgs.mutable_data();
        for (int i=0; i<gr.state.size(); i++)
            md[i] = gr.state[i].norm;

        auto pgr = PyGymResult(pgs,
            gr.n_errors,
            gr.dmgDealt,
            gr.dmgReceived,
            gr.ended,
            gr.victory
        );

        LOG("[pycb] Call wpycb(pgr)");
        wpycb(pgr);
        LOG("[pycb] return");
    };


    // Create PyCBInit from the given WPyCBInit
    const MMAI::PyCBInit pycbinit = [&wpycbinit](MMAI::CppCB &cppcb) {
        // Create WCppCB from the given CppCB
        WCppCB wcppcb = [cppcb](PyGymAction pga) {
            //           ^^^^^
            //           NOT a reference!
            // The reference is to a local var within AISimulator::init
            // and becomes dangling as soon as AISimulator::init returns
            // (as soon as wpycbinit returns)
            LOG("[wcppcb] start");

            // NOTE: we already have GIL, cppcb is called from python

            LOG("[wcppcb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[wcppcb] Convert pga -> ga");
            auto ga = static_cast<MMAI::GymAction>(pga);

            LOG("[wcppcb] Call cppcb(ga)");
            cppcb(ga);
            LOG("[wcppcb] return");
        };

        // NOTE: this may seem to work withoug GIL, but it is
        //       actually required (I once got an explicit error about it)
        LOG("[pycbinit] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[pycbinit] call wpycbinit(wcppcb)");
        wpycbinit(wcppcb);
        LOG("[pycbinit] return");
    };


    // Create PyCBSysInit from the given WPyCBSysInit
    // see pycbinit notes about GIL and lambda var scoping
    const MMAI::PyCBSysInit pycbsysinit = [&wpycbsysinit](MMAI::CppSysCB &cppsyscb) {
        // Create WCppSysCB from the given CppSysCB
        WCppSysCB wcppsyscb = [cppsyscb](std::string action) {
            LOG("[wcppsyscb] start");

            LOG("[wcppsyscb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[wcppsyscb] call cppsyscb");
            cppsyscb(action);
            LOG("[wcppsyscb] return");
        };

        LOG("[pycbsysinit] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[pycbsysinit] call wpycbsysinit(wcppsyscb)");
        wpycbsysinit(wcppsyscb);

        // Release GIL here instead of before calling start_vcmi()
        // (PySysCBInit is invoked in the main thread!)
        LOG("[pycbsysinit] release Python GIL");
        py::gil_scoped_release release;
        LOG("[pycbsysinit] return");
    };

    // Convert WPyCBResetInit -> PyCBResetInit
    const MMAI::PyCBResetInit pycbresetinit = [&wpycbresetinit](MMAI::CppResetCB &cppresetcb) {
        // Convert CppResetCBInit -> WCppResetCBInit
        WCppResetCB wcppresetcb = [cppresetcb]() {
            LOG("[wcppresetcb] start");

            LOG("[wcppresetcb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[wcppresetcb] call cppresetcb");
            cppresetcb();
            LOG("[wcppresetcb] return");
        };

        LOG("[pycbresetinit] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[pycbresetinit] call wpycbresetinit(wcppresetcb)");
        wpycbresetinit(wcppresetcb);
        LOG("[pycbresetinit] return");
    };

    // NOTE: GIL is released when CPP calls the sysinit callback
    // LOG("release Python GIL");
    // py::gil_scoped_release release;

    auto cbprovider = MMAI::CBProvider{pycbresetinit, pycbsysinit, pycbinit, pycb};

    // TODO: config values
    // std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";

    LOG("Start VCMI");
    start_vcmi(mapname, cbprovider);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}


void start(
    const WPyCBResetInit &wpycbresetinit,
    const WPyCBSysInit &wpycbsysinit,
    const WPyCBInit &wpycbinit,
    const WPyCB &wpycb,
    const std::string mapname
) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start (main thread)");

    // TODO: read config
    std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";

    // This must happen in the main thread (SDL requires it)
    LOG("call preinit_vcmi(...)");
    preinit_vcmi(resdir);

    LOG("launch new thread");
    boost::thread t([wpycbresetinit, wpycbsysinit, wpycbinit, wpycb, mapname]() {
        _start(wpycbresetinit, wpycbsysinit, wpycbinit, wpycb, mapname);
    });

    LOG("return");
}

PYBIND11_MODULE(cppconnector, m) {
    m.def("start", &start, "Start VCMI");
    m.def("get_state_size", &get_state_size, "Get number of elements in state");
    m.def("get_action_max", &get_action_max, "Get max expected value of action");

    py::class_<PyGymResult>(m, "PyGymResult")
        .def("n_errors", &PyGymResult::n_errors)
        .def("state", &PyGymResult::state)
        .def("dmg_dealt", &PyGymResult::dmg_dealt)
        .def("dmg_received", &PyGymResult::dmg_received)
        .def("is_battle_over", &PyGymResult::is_battle_over)
        .def("is_victorious", &PyGymResult::is_victorious);
}
