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

// use long names to avoid symbol collisions

extern "C" __attribute__((visibility("default"))) void ConnectorLoader_init(std::string path);
extern "C" __attribute__((visibility("default"))) MMAI::Export::Action ConnectorLoader_getAction(const MMAI::Export::Result* r);
