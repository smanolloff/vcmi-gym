#include <memory>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/chrono.hpp>
#include "client_simulator.h"

void ClientSimulator::start(const PyCBInit &pycbinit, const PyCB &pycb) {
    LOG("called");

    LOG("acquire Python GIL");
    py::gil_scoped_acquire acquire;

    // XXX: for some reason, this segfaults without a GIL
    // maybe because it *stores* the pycbinit functions?
    LOG("this->ai = AISimulator(pycbinit, pycb)");
    ai = std::make_shared<AISimulator>(pycbinit, pycb);

    LOG("release Python GIL");
    py::gil_scoped_release release;

    for(int i=0; ; i++) {
        LOGSTR("calling ai->activeStack(", std::to_string(i) + ")");
        ai->activeStack(i);
    }

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}
