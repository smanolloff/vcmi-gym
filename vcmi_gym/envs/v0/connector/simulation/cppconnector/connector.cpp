#include <pybind11/functional.h>

#include "connector.h"
#include "server_simulator.h"

void start_vcmi(const PyCBInit &pycbinit, const PyCB &pycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    LOG("release Python GIL");
    py::gil_scoped_release release;

    LOG("Start server");
    ServerSimulator().start(pycbinit, pycb);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}

PYBIND11_MODULE(connsimulator, m) {
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
