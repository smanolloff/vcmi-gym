#include <memory>

#include "common.h"
#include "client_simulator.h"

class ServerSimulator {
  std::shared_ptr<ClientSimulator> client;

public:
  void start(WPyCBInit pycbinit, WPyCB pycb);
};

