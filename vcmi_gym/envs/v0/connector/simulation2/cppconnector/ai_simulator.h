#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include "common.h"

class AISimulator {
  // simulate common.h NOT included (we still need it for LOG macro tho)
  // simulate a totally different project

  // WPyCBInit
  const std::function<void(std::function<void(const float (&arr)[3])>)> &pycbinit;

  // WCppCB
  const std::function<void(const float (&arr)[3])> pycb;

  bool inited;
  float action[3];
  boost::mutex m;
  boost::condition_variable cond;

  void init();
  std::string aryToStr(const float (&ary)[3]);

public:
  AISimulator(
    const std::function<void(std::function<void(const float (&arr)[3])>)> &pycbinit, // WPyCBInit
    const std::function<void(const float (&arr)[3])> &pycb // WPyCB
  );

  void activeStack(int i);
  void cppcb(const float (&arr)[3]);
};

