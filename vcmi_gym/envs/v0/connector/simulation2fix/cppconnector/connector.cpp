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
    const WPyCB wpycb = [&pycb]() {
        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;
        pycb();
    };

    // THIS ONE!!!!!!
    // prevents wcppcb below from going out-of-scope as soon as
    // wpycbinit exitsby keeping one more reference to it?!?
    std::function<void()> saver;

    // Convert PyCBInit -> WPyCBInit
    const WPyCBInit wpycbinit = [&pycbinit, &saver](WCppCB &wcppcb) {
        // LOG("AAAAASADSDADSDASSDADASDASADSDASDSA call wcppcb(float*)");
        // const float bbb[3] = {66.0f, 66.0f, 66.0f};
        // wcppcb(bbb);

        saver = wcppcb;
        LOG("acquire Python GIL");
        py::gil_scoped_acquire acquire;

        // Convert WCppCBInit -> CppCBInit
        CppCB cppcb = [&wcppcb]() {
            // this is called by python -- we already have the GIL
            // convert py::array_t<float> to float*
            LOG("acquire Python GIL");
            py::gil_scoped_acquire acquire;

            boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
            // This does not work
            // Here we're in the main python2.py thread (before spawning connector)
            if (!wcppcb)
                LOG("NUUUUULLLLLLL");

            LOG("!!!!!!!! CALL cppcb(...)");
            // saver();

            //
            // NOTE:
            // a.getB() returns a const value
            // However, assigning it explicitly to a non-const var
            // effectively drops the const modifier
            // To make it on a single line:
            // const float * data = const_cast<py::array_t<float>&>(a.getB()).mutable_data();
            //
            // py::array_t<float> b = a.getB();
            // const float * data = b.mutable_data();

            // LOG("release Python GIL");
            // py::gil_scoped_release release;
            // LOG("acquire Python GIL");
            // py::gil_scoped_acquire acquire;
            // boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));

            // LOG("call wcppcb(float*)");
            // const float bbb[3] = {66.0f, 66.0f, 66.0f};

            // wcppcb(bbb);
        };

        // This works - maybe because we-re in the AI thread?
        LOG("!!!!!!!! CALL cppcb(...)");
        wcppcb();
        cppcb();

        LOG("call pycbinit(cppcb)");
        pycbinit(cppcb);
        // boost::this_thread::sleep_for(boost::chrono::milliseconds(5000));
        LOG("RETURN");
    };


    LOG("release Python GIL");
    py::gil_scoped_release release;


    LOG("Start server");
    ServerSimulator().start(wpycbinit, wpycb);

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}

PYBIND11_MODULE(connsimulator2fix, m) {
    m.def("start_vcmi", &start_vcmi, "Start VCMI");
}
