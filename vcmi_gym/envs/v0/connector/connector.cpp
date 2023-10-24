#include <cstdio>
// #include <boost/thread.hpp>
#include <pybind11/functional.h>
#include "myclient.h"

void cpp_function(const std::function<void(int)> &callback, int value) {
    printf(">>> in connector.cpp; %d; callback(%d)\n", value, value+1);

    // ideally, callback should be passed to "run", then
    // further upstream all the way to:
    // 1. the AI initializer ...
    // 2. the AI initializer
    // which should call it as soon as BattleInit is called
    // boost::thread thread([]() {
        // printf(">>> runner thread starting...\n");
        // std::string respath = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";
        // mymain(respath, callback);
        mymain(4);
    // });

    // Perform some C++ work
    // Call the Python callback function
    printf(">>> runner thread started.\n");
}

PYBIND11_MODULE(connector, m) {
    m.def("cpp_function", &cpp_function, "Call C++ function with a Python callback");
}
