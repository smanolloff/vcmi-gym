#include <cstdio>
#include <dlfcn.h>

#include "connload.h"

int main() {
    void* handle = dlopen("build/libconnload.dylib", RTLD_LAZY);
    if (!handle) {
        printf("Error loading the library: %s\n", dlerror());
        return 1;
    }

    // ie. "fn = getResult"
    auto f_getAction = reinterpret_cast<decltype(&getAction)>(dlsym(handle, "getAction"));
    if (!f_getAction) {
        printf("Error getting the f_getAction: %s\n", dlerror());
        dlclose(handle); // Close the library
        return 1;
    }


    printf("IN MAIN\n");
    auto state = MMAI::Export::State{};

    for (int i=0; i<state.size(); i++)
        state[i] = MMAI::Export::NValue(i, 0, state.size()-1);

    auto actmask = MMAI::Export::ActMask{};
    for (int i=0; i < actmask.size(); i++) {
        actmask[i] = (i % 2 == 0);
    }

    auto result = MMAI::Export::Result(state, actmask, 0, 0, 0, 0, 0, 0);

    return 0;
}
