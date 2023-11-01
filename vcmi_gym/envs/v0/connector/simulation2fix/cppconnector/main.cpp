#include <boost/thread.hpp>
#include "connector.h"
#include "common.h"

struct Container {
  CppCB cppcb;
};

int main() {
  LOG("start")

  Container container{};

  PyCBInit pycbinit = [&container](CppCB cppcb) {
    LOG("** PYCBINIT SIMULATE **");
    container.cppcb = cppcb;
  };

  PyCB pycb = [&container](State s) {
    LOG("** PYCB SIMULATE **");
    Action a{"putka"};
    container.cppcb(a);
  };

  boost::thread([&pycbinit, &pycb]() { start_vcmi(pycbinit, pycb); });

  LOG("Entering sleep loop...");
  while(true)
      boost::this_thread::sleep_for(boost::chrono::seconds(1));
}
