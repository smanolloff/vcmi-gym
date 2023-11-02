#pragma once

#include <array>

using ActionF = std::array<float, 3>;
using StateF = std::array<float, 3>;

// CppCB is a CPP function given to the GymEnv via PyCBSysInit (see below)
// GymEnv will invoke it on every "reset()" or "close()" calls
using CppSysCB = const std::function<void(const ActionF &arr)>;

// PyCBSysInit is a Python function passed to VCMI entrypoint.
// VCMI will invoke it once, with 1 argument: a CppSysCB (see above)
using PyCBSysInit = const std::function<void(CppSysCB)>;

// CppCB is a CPP function given to the GymEnv via PyCBInit (see below)
// GymEnv will invoke it on every "step()" call, with 1 argument: an ActionF
using CppCB = const std::function<void(const ActionF &arr)>;

// PyCBInit is a Python function passed to the AI constructor.
// AI constructor will invoke it once, with 1 argument: a CppCB (see above)
using PyCBInit = const std::function<void(CppCB)>;

// PyCB is a Python function passed to the AI constructor.
// AI will invoke it on every "yourTurn()" call, with 1 argument: a StateF
using PyCB = const std::function<void(const StateF &arr)>;

struct CBProvider {
    CBProvider(const PyCBInit pycbinit_, const PyCB pycb_)
    : pycbinit(pycbinit_), pycb(pycb_) {}

    const PyCBSysInit pycbsysinit;
    const PyCBInit pycbinit;
    const PyCB pycb;
};
