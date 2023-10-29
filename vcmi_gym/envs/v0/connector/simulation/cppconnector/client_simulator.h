#include <memory>

#include "common.h"
#include "ai_simulator.h"

class ClientSimulator {
  std::shared_ptr<AISimulator> ai;

public:
  void start(const PyCBInit &pycbinit, const PyCB &pycb);
};

