#include "main.h"
#include "pyclient.h" // "vendor" header file

int main() {
    // Convert WPyCB -> PyCB
    const MMAI::PyCB pycb = [](const MMAI::GymState &gymstate) {
        LOG("pycb called");
    };

    // Convert WPyCBInit -> PyCBInit
    const MMAI::PyCBInit pycbinit = [](MMAI::CppCB &cppcb) {
        LOG("pycbinit called");
    };


    // Convert WPyCBInit -> PyCBInit
    const MMAI::PyCBSysInit pycbsysinit = [](MMAI::CppSysCB &cppsyscb) {
        LOG("pycbsysinit called");
    };

    auto cbprovider = MMAI::CBProvider{pycbsysinit, pycbinit, pycb};

    // TODO: config values
    std::string resdir = "/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/vcmi/build/bin";
    std::string mapname = "simotest.vmap";

    LOG("Start VCMI");
    preinit_vcmi(resdir);

    // WTF for some reason linker says this is undefined symbol WTF
    start_vcmi(mapname, cbprovider);

    return 0;
}
