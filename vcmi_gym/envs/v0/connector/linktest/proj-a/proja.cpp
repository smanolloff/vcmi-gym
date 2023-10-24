#include <functional>
#include "proja.h"

int mymain(int i) {
  std::function<void(int)> callback;
  return i+1;
}
