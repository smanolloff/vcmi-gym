#include <pybind11/functional.h>

#include "cppconnector.h"
#include "pyclient.h" // "vendor" header file

// states are set in CPP and read by python
// => need to convert GymState to py::array_t
py::array_t<float> stateForPython(GymState gymstate) {
    // convert float* to py::array_t<float>
    auto pyary = py::array_t<float>(gymstate.size());
    auto md = pyary.mutable_data();

    for (int i=0; i<gymstate.size(); i++)
        md[i] = gymstate[i];

    return pyary;
}

// actions are set in python and read by CPP
// => need to convert py::array_t to ActionF
ActionF actionFromPython(py::array_t<float> pyaction) {
    ActionF actionf;
    auto data = pyaction.data();
    auto size = pyaction.size();
    assert(size == std::size(pyaction));

    for (int i=0; i<size; i++)
        actionf[i] = data[i];

    return actionf;
}

void _start(const WPyCBSysInit &wpycbsysinit, const WPyCBInit &wpycbinit, const WPyCB &wpycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    LOG("create wrappers around pycbinit, pycb")

    // Convert WPyCB -> PyCB
    const PyCB pycb = [&wpycb](const GymState &gymstate) {
        LOG("start");

        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("Create state(\"kur\", gymstate)");
        State state("kur", gymstate);

        LOG("Call wpycb(state)");
        wpycb(state);
    };


    // Convert WPyCBInit -> PyCBInit
    const PyCBInit pycbinit = [&wpycbinit](CppCB &cppcb) {
        // Convert WCppCBInit -> CppCBInit
        WCppCB wcppcb = [cppcb](Action a) {
            //           ^^^^^
            //           NOT a reference!
            // The reference is to a local var within AISimulator::init
            // and becomes dangling as soon as AISimulator::init returns
            // (as soon as wpycbinit returns)
            LOG("start");

            // NOTE: we already have GIL, cppcb is called from python

            LOG("release Python GIL");
            py::gil_scoped_release release;

            cppcb(a.action);
        };

        // NOTE: this may seem to work withoug GIL, but it is
        //       actually required (I once got an explicit error about it)
        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("call pycbinit(cppcb)");
        wpycbinit(wcppcb);

        LOG("RETURN");
    };


    // Convert WPyCBInit -> PyCBInit
    const PyCBSysInit pycbsysinit = [&wpycbsysinit](CppSysCB &cppsyscb) {
        // Convert WCppSysCBInit -> CppSysCBInit
        WCppSysCB wcppsyscb = [cppsyscb](std::string action) {
            //                 ^^^^^
            //                 NOT a reference!
            // The reference is to a local var within AISimulator::init
            // and becomes dangling as soon as AISimulator::init returns
            // (as soon as wpycbinit returns)
            LOG("start");

            // NOTE: we already have GIL, cppcb is called from python

            LOG("release Python GIL");
            py::gil_scoped_release release;

            cppsyscb(action);
        };


        // NOTE: this may seem to work withoug GIL, but it is
        //       actually required (I once got an explicit error about it)
        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        // TODO: implement the pycbsysinit in PYthon and pass it to start_connector
        LOG("call pycbinit(cppcb)");
        wpycbsysinit(wcppsyscb);

        // Release GIL here instead of before calling start_vcmi()
        LOG("release Python GIL");
        py::gil_scoped_release release;

        LOG("RETURN");
    };

    // NOTE: GIL is released when CPP calls the sysinit callback
    // LOG("release Python GIL");
    // py::gil_scoped_release release;

    auto cbprovider = CBProvider{pycbsysinit, pycbinit, pycb};

    // TODO: config values
    // std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";
    std::string mapname = "simotest.vmap";

    LOG("Start VCMI");
    // start_vcmi(resdir, mapname, cbprovider);
    start_vcmi(mapname, cbprovider);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}

void start(const WPyCBSysInit &wpycbsysinit, const WPyCBInit &wpycbinit, const WPyCB &wpycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start (main thread)");

    // TODO: read config
    std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";

    // This must happen in the main thread (SDL requires it)
    LOG("call preinit_vcmi(...)");
    preinit_vcmi(resdir);

    LOG("launch new thread");
    boost::thread([wpycbsysinit, wpycbinit, wpycb]() {
        _start(wpycbsysinit, wpycbinit, wpycb);
    });

    LOG("return");
}

PYBIND11_MODULE(conntest, m) {
    m.def("start", &start_connector, "Start VCMI");

    py::class_<Action>(m, "Action")
        .def(py::init<std::string, py::array_t<float>>());

    py::class_<State>(m, "State")
        .def(py::init<std::string, GymState>())
        .def("getStr", &State::getStr)
        .def("getState", &State::getState);}
