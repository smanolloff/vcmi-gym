#include <memory>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/chrono.hpp>
#include "client_simulator.h"

void ClientSimulator::start(PyCBInit &pycbinit, PyCB &pycb) {
    LOG("called");

    LOG("this->ai = AISimulator(pycbinit, pycb)");
    ai = std::make_shared<AISimulator>(pycbinit, pycb);

    for(int i=0; ; i++) {
        LOGSTR("calling ai->activeStack(", std::to_string(i) + ")");
        ai->activeStack(i);
    }

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}
