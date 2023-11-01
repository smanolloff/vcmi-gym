#include <memory>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/chrono.hpp>
#include "client_simulator.h"

void ClientSimulator::start(WPyCBInit wpycbinit, WPyCB wpycb) {
    LOG("called");

    LOG("this->ai = AISimulator(wpycbinit, wpycb)");
    ai = std::make_shared<AISimulator>(wpycbinit, wpycb);

    for(int i=0; ; i++) {
        LOGSTR("calling ai->activeStack(", std::to_string(i) + ")");
        ai->activeStack(i);
    }

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}
