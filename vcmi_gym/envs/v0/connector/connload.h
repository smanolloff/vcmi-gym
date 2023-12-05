#pragma once

#include "mmai_export.h" // "vendor" header file

extern "C" {
    __attribute__((visibility("default"))) MMAI::Export::Action getAction(MMAI::Export::Result* r);
}
