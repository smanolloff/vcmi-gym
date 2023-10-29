#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include "common.h"

class AISimulator {
  PyCBInit pycbinit;
  PyCB pycb;
  bool inited;
  Action action;
  boost::mutex m;
  boost::condition_variable cond;

  void init();

public:
  AISimulator(const PyCBInit &pycbinit, const PyCB &pycb);
  void activeStack(int i);
  void cppcb(const Action action);
};

