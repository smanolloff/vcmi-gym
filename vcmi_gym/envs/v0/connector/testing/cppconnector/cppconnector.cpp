#include <pybind11/functional.h>

#include "cppconnector.h"
#include "pyclient.h" // "vendor" header file

// NOT USED: GymAction is a uint16_t, not a np array now
// // actions are set in python and read by CPP
// // => need to convert py::array_t to GymAction
// GymAction actionFromPython(py::array_t<float> pyaction) {
//     GymAction gymaction;
//     auto data = pyaction.data();
//     auto size = pyaction.size();
//     assert(size == std::size(pyaction));

//     for (int i=0; i<size; i++)
//         gymaction[i] = data[i];

//     return gymaction;
// }

void _start_connector(const WPyCBSysInit &wpycbsysinit, const WPyCBInit &wpycbinit, const WPyCB &wpycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    LOG("create wrappers around pycbinit, pycb")

    // Convert WPyCB -> PyCB
    const MMAI::PyCB pycb = [&wpycb](const MMAI::GymState &gymstate) {
        LOG("[pycb] start");

        LOG("[pycb] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("[pycb] Convert gs -> pgs");
        auto pgs = PyGymState(gymstate.size());
        auto md = pgs.mutable_data();
        for (int i=0; i<gymstate.size(); i++)
            md[i] = gymstate[i];

        LOG("[pycb] Call wpycb(pgs)");
        wpycb(pgs);
        LOG("[pycb] return");
    };


    // Convert WPyCBInit -> PyCBInit
    const MMAI::PyCBInit pycbinit = [&wpycbinit](MMAI::CppCB &cppcb) {
        // Convert WCppCBInit -> CppCBInit
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

        LOG("[pycbinit] call pycbinit(cppcb)");
        wpycbinit(wcppcb);
        LOG("[pycbinit] return");
    };


    // Convert WPyCBInit -> PyCBInit
    const MMAI::PyCBSysInit pycbsysinit = [&wpycbsysinit](MMAI::CppSysCB &cppsyscb) {
        // Convert WCppSysCBInit -> CppSysCBInit
        WCppSysCB wcppsyscb = [cppsyscb](std::string action) {
            //                 ^^^^^
            //                 NOT a reference!
            // The reference is to a local var within AISimulator::init
            // and becomes dangling as soon as AISimulator::init returns
            // (as soon as wpycbinit returns)
            LOG("[wcppsyscb] start");

            // NOTE: we already have GIL, cppcb is called from python

            LOG("[wcppsyscb] release Python GIL");
            py::gil_scoped_release release;

            LOG("[wcppsyscb] call cppsyscb");
            cppsyscb(action);
            LOG("[wcppsyscb] return");
        };


        // NOTE: this may seem to work withoug GIL, but it is
        //       actually required (I once got an explicit error about it)
        LOG("[pycbsysinit] acquire Python GIL");
        py::gil_scoped_acquire acquire;

        // TODO: implement the pycbsysinit in PYthon and pass it to start_connector
        LOG("[pycbsysinit] call pycbinit(cppcb)");
        wpycbsysinit(wcppsyscb);

        // Release GIL here instead of before calling start_vcmi()
        LOG("[pycbsysinit] release Python GIL");
        py::gil_scoped_release release;
        LOG("[pycbsysinit] return");
    };

    // NOTE: GIL is released when CPP calls the sysinit callback
    // LOG("release Python GIL");
    // py::gil_scoped_release release;

    auto cbprovider = MMAI::CBProvider{pycbsysinit, pycbinit, pycb};

    // TODO: config values
    // std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";
    std::string mapname = "simotest.vmap";

    LOG("Start VCMI");
    // start_vcmi(resdir, mapname, cbprovider);
    start_vcmi(mapname, cbprovider);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}

void start_connector(const WPyCBSysInit &wpycbsysinit, const WPyCBInit &wpycbinit, const WPyCB &wpycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start (main thread)");

    // TODO: read config
    std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";

    // This must happen in the main thread (SDL requires it)
    LOG("call preinit_vcmi(...)");
    preinit_vcmi(resdir);

    LOG("launch new thread");
    boost::thread([wpycbsysinit, wpycbinit, wpycb]() {
        _start_connector(wpycbsysinit, wpycbinit, wpycb);
    });

    LOG("return");
}

PYBIND11_MODULE(conntest, m) {
    m.def("start_connector", &start_connector, "Start VCMI");
}
