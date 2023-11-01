#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include "common.h"

class AISimulator {
  // simulate common.h NOT included (we still need it for LOG macro tho)
  // simulate a totally different project

  // WPyCBInit
  std::function<void(std::function<void(float * arr)>)> pycbinit;

  // WCppCB
  std::function<void(float * arr)> pycb;

  bool inited;
  float * action;
  boost::mutex m;
  boost::condition_variable cond;

  void init();
  std::string aryToStr(float * &ary);

public:
  AISimulator(
    std::function<void(std::function<void(float * arr)>)> pycbinit, // WPyCBInit
    std::function<void(float * arr)> pycb // WPyCB
  );

  void activeStack(int i);
  void cppcb(float * arr);
};

