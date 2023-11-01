#include <pybind11/functional.h>

#include "connector.h"
#include "server_simulator.h"

void start_vcmi(const PyCBInit &pycbinit, const PyCB &pycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    LOG("create wrappers around pycbinit, pycb")

    // XXX: do we need GIL when capturing Py defs in a [] block?
    //      Will try without GIL first.

    // Convert PyCB -> WPyCB
    const WPyCB wpycb = [&pycb](const StateF &data) {
        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        // convert float* to py::array_t<float>
        auto pyary = py::array_t<float>(3);
        auto md = pyary.mutable_data();
        std::string s = "";

        for (int i=0; i<std::size(data); i++) {
            md[i] = data[i];
            s += std::string(" ") + std::to_string(data[i]);
        }

        pycb(State{s, pyary});
    };


    // Convert PyCBInit -> WPyCBInit
    const WPyCBInit wpycbinit = [&pycbinit](WCppCB &wcppcb) {
        // Convert WCppCBInit -> CppCBInit
        CppCB cppcb = [wcppcb](Action a) {
            //         ^^^^^^
            //         NOT a reference!
            // The reference is to a local var within AISimulator::init
            // and becomes dangling as soon as AISimulator::init returns
            // (as soon as wpycbinit returns)

            //
            // NOTE:
            // a.getB() returns a const value
            // However, assigning it explicitly to a non-const var
            // effectively drops the const modifier
            // To make it on a single line:
            // const float * data = const_cast<py::array_t<float>&>(a.getB()).mutable_data();
            //

            // NOTE: using b.data()  may allow us to use a.getB().data()
            py::array_t<float> b = a.getB();
            const float * data = b.data();

            ActionF actionf;
            for (int i=0; i<b.size(); i++)
                actionf[i] = data[i];

            LOG("release Python GIL");
            py::gil_scoped_release release;

            wcppcb(actionf);
        };

        // This works - maybe because we-re in the AI thread?
        // LOG("!!!!!!!! CALL cppcb(...)");
        // wcppcb(static_cast<const float*>(std::array<float, 3>{66.0f, 66.0f, 66.0f}.data()));

        LOG("call pycbinit(cppcb)");
        pycbinit(cppcb);
        LOG("RETURN");
    };


    LOG("release Python GIL");
    py::gil_scoped_release release;


    LOG("Start server");
    ServerSimulator().start(wpycbinit, wpycb);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}

PYBIND11_MODULE(connsimulator2, m) {
    m.def("start_vcmi", &start_vcmi, "Start VCMI");

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def("setA", &Action::setA)
        .def("getA", &Action::getA)
        .def("setB", &Action::setB)
        .def("getB", &Action::getB);

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def("setA", &State::setA)
        .def("getA", &State::getA)
        .def("setB", &State::setB)
        .def("getB", &State::getB);
}
