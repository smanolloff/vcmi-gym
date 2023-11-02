#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include "common.h"

class AISimulator {
  // simulate common.h NOT included (we still need it for LOG macro tho)
  // simulate a totally different project

  // WPyCBInit
  const std::function<void(std::function<void(const std::array<float, 3>)>)> pycbinit;

  // WCppCB
  const std::function<void(const std::array<float, 3>)> pycb;

  bool inited;
  std::array<float, 3> action;
  boost::mutex m;
  boost::condition_variable cond;

  void init();
  std::string aryToStr(const std::array<float, 3>);

  std::function<void(std::function<void(const std::array<float, 3>)>)> loadPyCbInit(std::any cbprovider);
  std::function<void(const std::array<float, 3>)> loadPyCb(std::any cbprovider);

public:
  AISimulator(const std::any cbprovider);

  void activeStack(int i);
  void cppcb(const std::array<float, 3>);
};

