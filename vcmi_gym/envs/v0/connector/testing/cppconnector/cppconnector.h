#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include "common.h"

void preinit_vcmi();

void start_connector(
  const WPyCBSysInit &wpycbsysinit,
  const WPyCBInit &wpycbinit,
  const WPyCB &wpycb
);
