#include "main.h"
#include "pyclient.h" // "vendor" header file

int main() {
    // Convert WPyCB -> PyCB
    const PyCB pycb = [](const StateF &statef) {
        LOG("pycb called");
    };

    // Convert WPyCBInit -> PyCBInit
    const PyCBInit pycbinit = [](CppCB &cppcb) {
        LOG("pycbinit called");
    };


    // Convert WPyCBInit -> PyCBInit
    const PyCBSysInit pycbsysinit = [](CppSysCB &cppsyscb) {
        LOG("pycbsysinit called");
    };

    auto cbprovider = CBProvider{pycbsysinit, pycbinit, pycb};

    // TODO: config values
    std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";
    std::string mapname = "simotest.vmap";

    LOG("Start VCMI");
    preinit_vcmi(resdir);

    // WTF for some reason linker says this is undefined symbol WTF
    start_vcmi(mapname, cbprovider);

    return 0;
}
