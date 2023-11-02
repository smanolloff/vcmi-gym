#include <pybind11/functional.h>

#include "connector.h"
#include "server_simulator.h"

// states are set in CPP and read by python
// => need to convert StateF to py::array_t
py::array_t<float> stateForPython(StateF statef) {
    // convert float* to py::array_t<float>
    auto pyary = py::array_t<float>(statef.size());
    auto md = pyary.mutable_data();

    for (int i=0; i<statef.size(); i++)
        md[i] = statef[i];

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

void start_vcmi(const WPyCBInit &wpycbinit, const WPyCB &wpycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    LOG("create wrappers around pycbinit, pycb")

    // XXX: do we need GIL when capturing Py defs in a [] block?
    //      Will try without GIL first.

    // Convert WPyCB -> PyCB
    const PyCB pycb = [&wpycb](const StateF &statef) {
        LOG("start");

        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        LOG("Create state(\"kur\", statef)");
        State state("kur", statef);

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

        // This works - maybe because we-re in the AI thread?
        // LOG("!!!!!!!! CALL cppcb(...)");
        // wcppcb(static_cast<const float*>(std::array<float, 3>{66.0f, 66.0f, 66.0f}.data()));

        LOG("call pycbinit(cppcb)");
        wpycbinit(wcppcb);
        LOG("RETURN");
    };


    LOG("release Python GIL");
    py::gil_scoped_release release;


    LOG("Start server");
    ServerSimulator().start(pycbinit, pycb);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}

PYBIND11_MODULE(connsimulator, m) {
    m.def("start_vcmi", &start_vcmi, "Start VCMI");

    py::class_<Action>(m, "Action")
        .def(py::init<std::string, py::array_t<float>>());

    py::class_<State>(m, "State")
        .def(py::init<std::string, StateF>())
        .def("getStr", &State::getStr)
        .def("getState", &State::getState);}
