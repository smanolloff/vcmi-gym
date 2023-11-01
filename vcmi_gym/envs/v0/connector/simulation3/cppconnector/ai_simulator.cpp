#include "ai_simulator.h"

AISimulator::AISimulator(
    std::function<void(std::function<void(float * arr)>)> _pycbinit,
    std::function<void(float * arr)> _pycb)
    : pycbinit(_pycbinit), pycb(_pycb), inited(false) {
  LOG("constructor called");
}

std::string AISimulator::aryToStr(float * &ary) {
    return std::to_string(ary[0]) + " " + std::to_string(ary[1]) + " " + std::to_string(ary[2]);
}


// this can't be defined here as  as it will need access to
// a cond variable which it should modify
// TODO: what if the cond variable is global?
// State act(Action a) {
//     LOG("called with action.getA(): %s, action.getB(): ?\n", action.getA());
// }

void AISimulator::cppcb(float * action_ary) {
    LOGSTR("called with action_ary: ", aryToStr(action_ary));

    LOG("assign this->action = action_ary");
    action = action_ary;

    // here we would call cb->makeAction()
    // ...

    // DEBUG: uncomment this, just temporary commented to test sth
    // LOG("cond.notify_one()");
    // cond.notify_one();

    LOG("return");
}

// private
void AISimulator::init() {
    LOG("called");

    // XXX: Danger: SIGSERV?
    LOG("call this->pycbinit([this](float * action_ary) { this->cppcb(action_ary) })");
    this->pycbinit([this](float * action_ary) { this->cppcb(action_ary); });

    LOG("set inited = true");
    inited = true;

    LOG("return");
}

void AISimulator::activeStack(int i) {
    LOG("called");

    if (!inited)
        init();

    LOG("sleep 100ms");
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));

    float state_ary[3] = {1.0f, 2.0f, 3.0f};

    LOG("acquire lock");
    boost::unique_lock<boost::mutex> lock(this->m);

    LOG("call this->pycb(state_ary)");
    this->pycb(state_ary);

    // We've set some events in motion:
    //  - in python, "env" now has our cppcb stored (via pycbinit)
    //  - in python, "env" now has the state stored (via pycb)
    //  - in python, "env" constructor can now return (pycb also set an event)
    //  - in python, env.step(action) will be called, which will call cppcb
    // our cppcb will then call AI->cb->makeAction()
    // ...we wait until that happens, and FINALLY we can return from yourTurn
    LOG("cond.wait()");
    this->cond.wait(lock);

    LOGSTR("this->action: ", aryToStr(action));
    LOG("return");
}
