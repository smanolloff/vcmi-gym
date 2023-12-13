#pragma once

/*****
****** THIS FILE LIVES IN:
******
****** connector/loader.h
******
*****/

#ifdef __LOADED_FROM_MMAI
#include "export.h"
#else
#include "mmai_export.h" // "vendor" header file
#endif

// internal
void init(MMAI::Export::Side side, std::string gymdir, std::string modelfile);
MMAI::Export::Action getAction(MMAI::Export::Side &side, const MMAI::Export::Result* &r);

// external
// use long names to avoid symbol collisions

extern "C" __attribute__((visibility("default"))) void ConnectorLoader_initAttacker(
    std::string gymdir,
    std::string modelfile
);

extern "C" __attribute__((visibility("default"))) void ConnectorLoader_initDefender(
    std::string gymdir,
    std::string modelfile
);

extern "C" __attribute__((visibility("default"))) MMAI::Export::Action ConnectorLoader_getActionAttacker(
    const MMAI::Export::Result* r
);

extern "C" __attribute__((visibility("default"))) MMAI::Export::Action ConnectorLoader_getActionDefender(
    const MMAI::Export::Result* r
);
