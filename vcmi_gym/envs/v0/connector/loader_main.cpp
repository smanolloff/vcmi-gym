// =============================================================================
// Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <cstdio>
#include <dlfcn.h>

#include "loader.h"
#include "mmai_export.h"

int main() {
    void* handle = dlopen("build/libloader.dylib", RTLD_LAZY);
    if (!handle) {
        printf("Error loading the library: %s\n", dlerror());
        return 1;
    }

    auto f_init = reinterpret_cast<decltype(&ConnectorLoader_initAttacker)>(dlsym(handle, "ConnectorLoader_initAttacker"));
    if (!f_init) {
        printf("Error getting the f_init: %s\n", dlerror());
        dlclose(handle); // Close the library
        return 1;
    }

    auto f_getAction = reinterpret_cast<decltype(&ConnectorLoader_getActionAttacker)>(dlsym(handle, "ConnectorLoader_getActionAttacker"));
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
    auto gympath = "/Users/simo/Projects/vcmi-gym";
    auto modelpath = "data/M8-PBT-MPPO-20231204_191243/576e9_00000/checkpoint_000139/model.zip";
    f_init(MMAI::Export::STATE_ENCODING_DEFAULT, gympath, modelpath);

    auto state = MMAI::Export::StateUnencoded{};

    for (int i=0; i<165; i++)
        for (int i=0; i<static_cast<int>(MMAI::Export::Attribute::_count); i++)
        state.at(i) = MMAI::Export::OneHot(MMAI::Export::Attribute(i));

    auto actmask = MMAI::Export::ActMask{};
    for (int i=0; i < actmask.size(); i++) {
        actmask[i] = (i % 2 == 0);
    }

    auto attnmasks = MMAI::Export::AttnMasks{};

    auto result = MMAI::Export::Result(state, actmask, attnmasks, MMAI::Export::Side(0), 0, 0, 0, 0, 0, 0, 0, 0);
    printf("IN MAIN: GOT ACTION: %d\n", f_getAction(&result));
    return 0;
}
