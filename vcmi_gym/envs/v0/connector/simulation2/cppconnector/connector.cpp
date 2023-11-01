#include <pybind11/functional.h>

#include "connector.h"
#include "server_simulator.h"

void start_vcmi(PyCBInit &pycbinit, PyCB &pycb) {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    LOG("create wrappers around pycbinit, pycb")

    // XXX: do we need GIL when capturing Py defs in a [] block?
    //      Will try without GIL first.

    // Convert PyCB -> WPyCB
    WPyCB wpycb = [&pycb](const float * data) {
        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        // convert float* to py::array_t<float>
        // TODO: magic number 3 should be a constant STATE_ARY_LENGTH
        auto pyary = py::array_t<float>(3);
        float * md = pyary.mutable_data();
        std::string s = "";
        for(int i=0; i < 3; i++) {
            md[i] = data[i];
            s += std::string(" ") + std::to_string(data[i]);
        }

        pycb(State{s, pyary});
    };

    // Convert PyCBInit -> WPyCBInit
    const WPyCBInit wpycbinit = [&pycbinit](WCppCB &wcppcb) {
        // LOG("AAAAASADSDADSDASSDADASDASADSDASDSA call wcppcb(float*)");
        // const float bbb[3] = {66.0f, 66.0f, 66.0f};
        // wcppcb(bbb);

        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        // Convert WCppCBInit -> CppCBInit
        const CppCB cppcb = [&wcppcb](Action a) {
            // this is called by python -- we already have the GIL
            // convert py::array_t<float> to float*
            LOG("acquire Python GIL");
            py::gil_scoped_acquire acquire;

            LOG("!!!!!!!! CALL cppcb(...)");
            wcppcb(static_cast<const float*>(std::array<float, 3>{66.0f, 66.0f, 66.0f}.data()));

            //
            // NOTE:
            // a.getB() returns a const value
            // However, assigning it explicitly to a non-const var
            // effectively drops the const modifier
            // To make it on a single line:
            // const float * data = const_cast<py::array_t<float>&>(a.getB()).mutable_data();
            //
            py::array_t<float> b = a.getB();
            const float * data = b.mutable_data();

            // LOG("release Python GIL");
            // py::gil_scoped_release release;
            // LOG("acquire Python GIL");
            // py::gil_scoped_acquire acquire;
            boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));

            LOG("call wcppcb(float*)");
            const float bbb[3] = {66.0f, 66.0f, 66.0f};

            if (!wcppcb)
                LOG("NUUUUULLLLLLL");

            wcppcb(bbb);
        };

        LOG("!!!!!!!! CALL cppcb(...)");
        wcppcb(static_cast<const float*>(std::array<float, 3>{66.0f, 66.0f, 66.0f}.data()));

        LOG("call pycbinit(cppcb)");
        pycbinit(cppcb);
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
