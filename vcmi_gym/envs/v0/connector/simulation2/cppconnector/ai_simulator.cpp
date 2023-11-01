#include "ai_simulator.h"

AISimulator::AISimulator(
    const std::function<void(std::function<void(const float (&arr)[3])>)> &_pycbinit,
    const std::function<void(const float (&arr)[3])> &_pycb)
    : pycbinit(_pycbinit), pycb(_pycb), inited(false) {
  LOG("constructor called");
}

std::string AISimulator::aryToStr(const float (&arr)[3]) {
    return std::to_string(arr[0]) + " " + std::to_string(arr[1]) + " " + std::to_string(arr[2]);
}


// this can't be defined here as const, as it will need access to
// a cond variable which it should modify
// TODO: what if the cond variable is global?
// const State act(Action a) {
//     LOG("called with action.getA(): %s, action.getB(): ?\n", action.getA());
// }

void AISimulator::cppcb(const float (&arr)[3]) {
    LOGSTR("called with arr: ", aryToStr(arr));

    // Need to copy - storing just the pointer is dangerous
    // as the underlying array may get wiped
    LOG("Copy arr to this->action");
    std::copy(std::begin(arr), std::end(arr), std::begin(action));

    // here we would call cb->makeAction()
    // ...

    LOG("cond.notify_one()");
    cond.notify_one();

    LOG("return");
}

// private
void AISimulator::init() {
    LOG("called");

    // XXX: Danger: SIGSERV?
    LOG("call this->pycbinit([this](const float (&arr)[3]) { this->cppcb(arr) })");
    this->pycbinit([this](const float (&arr)[3]) { this->cppcb(arr); });

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

    const float state_ary[3] = {1.0f, 2.0f, 3.0f};

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
