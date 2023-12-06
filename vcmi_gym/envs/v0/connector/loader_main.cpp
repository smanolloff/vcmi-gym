#include <cstdio>
#include <dlfcn.h>

#include "loader.h"

int main() {
    void* handle = dlopen("build/libloader.dylib", RTLD_LAZY);
    if (!handle) {
        printf("Error loading the library: %s\n", dlerror());
        return 1;
    }
    std::function<MMAI::Export::Action(MMAI::Export::Result* r)> f_getAction;

    auto f_init = reinterpret_cast<decltype(&ConnectorLoader_init)>(dlsym(handle, "ConnectorLoader_init"));
    if (!f_init) {
        printf("Error getting the f_init: %s\n", dlerror());
        dlclose(handle); // Close the library
        return 1;
    }

    f_getAction = reinterpret_cast<decltype(&ConnectorLoader_getAction)>(dlsym(handle, "ConnectorLoader_getAction"));
    if (!f_getAction) {
        printf("Error getting the f_getAction: %s\n", dlerror());
        dlclose(handle); // Close the library
        return 1;
    }

    // Not needed (couldn't implement a proper shutdown anyway)
    // re-loading the dylib works fine
    // auto f_shutdown = reinterpret_cast<decltype(&ConnectorLoader_shutdown)>(dlsym(handle, "ConnectorLoader_shutdown"));
    // if (!f_shutdown) {
    //     printf("Error getting the f_shutdown: %s\n", dlerror());
    //     dlclose(handle); // Close the library
    //     return 1;
    // }


    printf("IN MAIN\n");
    auto path = "/Users/simo/Projects/vcmi-gym/data/M8-PBT-MPPO-20231204_191243/576e9_00000/checkpoint_000139/model.zip";
    f_init(path);

    auto state = MMAI::Export::State{};

    for (int i=0; i<state.size(); i++)
        state[i] = MMAI::Export::NValue(i, 0, state.size()-1);

    auto actmask = MMAI::Export::ActMask{};
    for (int i=0; i < actmask.size(); i++) {
        actmask[i] = (i % 2 == 0);
    }

    auto result = MMAI::Export::Result(state, actmask, 0, 0, 0, 0, 0, 0);
    printf("IN MAIN: GOT ACTION: %d\n", f_getAction(&result));
    printf("IN MAIN: GOT ACTION: %d\n", f_getAction(&result));
    printf("IN MAIN: GOT ACTION: %d\n", f_getAction(&result));
    return 0;
}
