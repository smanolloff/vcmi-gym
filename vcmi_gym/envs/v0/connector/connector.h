#pragma once

#include <filesystem>
#include <condition_variable>
#include <thread>
#include <cstdio>
#include <iostream>

#include "conncommon.h"

enum ConnectorState {
    NEW,
    AWAITING_ACTION,
    AWAITING_RESULT,
};

class Connector {
    std::mutex m1;
    std::mutex m2;
    std::condition_variable cond1;
    std::condition_variable cond2;

    ConnectorState state = ConnectorState::NEW;

    const std::string mapname;
    const std::string loglevelGlobal;
    const std::string loglevelAI;
    std::thread vcmithread;
    MMAI::Export::F_Sys f_sys;
    std::unique_ptr<MMAI::Export::CBProvider> cbprovider = std::make_unique<MMAI::Export::CBProvider>(nullptr);
    MMAI::Export::Action action;
    const MMAI::Export::Result * result;

    const P_Result convertResult(const MMAI::Export::Result * r);
    MMAI::Export::Action getAction(const MMAI::Export::Result * r);
    const MMAI::Export::Action getActionDummy(MMAI::Export::Result);

public:
    Connector(
        const std::string mapname,
        const std::string loglevelGlobal,
        const std::string loglevelAI
    );

    const P_Result start();
    const P_Result reset();
    const P_Result act(const MMAI::Export::Action a);
    const std::string renderAnsi();

    // Called when VcmiGym is started from within VCMI itself
    // (ie. VCMI is started normally, and vcmi-gym is started as its AI)
    const MMAI::Export::CBProvider* getCBProvider();
};
