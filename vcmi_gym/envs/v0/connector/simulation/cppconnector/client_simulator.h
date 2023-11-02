#include <memory>

#include "common.h"
#include "ai_simulator.h"

class ClientSimulator {
  std::shared_ptr<AISimulator> ai;

public:
  void start(std::any &baggage);
};

