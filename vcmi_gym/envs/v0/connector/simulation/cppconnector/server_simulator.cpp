#include <memory>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/chrono.hpp>
#include "server_simulator.h"

void ServerSimulator::start(const PyCBInit &pycbinit, const PyCB &pycb) {
    LOG("called");

    LOG("this->client = ClientSimulator()");
    client = std::make_shared<ClientSimulator>();

    // simulate starting VCMI client, which inits AI
    // no need of vcmi client simulator
    LOG("boost::thread(client->start(pycbinit, pycb))");
    boost::thread([this, pycbinit, pycb]() { client->start(pycbinit, pycb); });

    LOG("Entering sleep loop...");
    while(true)
        boost::this_thread::sleep_for(boost::chrono::seconds(1));

    LOG("return !!!! !SHOULD NEVER HAPPEN");
}
