#include "ai_simulator.h"

AISimulator::AISimulator(const PyCBInit &pycbinit, const PyCB &pycb) {
  LOG("constructor called");
  this->pycbinit = pycbinit;
  this->pycb = pycb;
  this->inited = false;
  LOG("return");
}

// this can't be defined here as const, as it will need access to
// a cond variable which it should modify
// TODO: what if the cond variable is global?
// const State act(Action a) {
//     LOG("called with action.getA(): %s, action.getB(): ?\n", action.getA());
// }

void AISimulator::cppcb(const Action a) {
    LOGSTR("called with a.getA()=", a.getA());

    // TODO: no need to acquire GIL here?
    // (caller is a Python thread)
    LOG("acquire lock");
    boost::lock_guard<boost::mutex> lock(m);

    LOG("assign action = action");
    action = a;

    LOG("cond.notify_one()");
    cond.notify_one();

    LOG("return");
}

// private
void AISimulator::init() {
    LOG("called");

    // Ensure we pass an already locked mutex to the cppcb!
    LOG("acquire lock");
    boost::unique_lock<boost::mutex> lock(this->m);

    // XXX: Danger: SIGSERV?
    LOG("call this->pycbinit([this](Action a) { this->cppcb(a) })");
    this->pycbinit([this](Action a) { this->cppcb(a); });

    LOG("set inited = true");
    inited = true;

    LOG("return");
}

void AISimulator::activeStack(int i) {
    LOG("called");

    LOG("acquire Python GIL");
    py::gil_scoped_acquire acquire;

    if (!inited)
        init();

    LOG("sleep 100ms");
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));

    // NOTE: GIL is required for constructing py::array_t objects
    auto ary = py::array_t<float>(3);
    float* data = ary.mutable_data();
    data[0] = i;
    data[1] = i;
    data[2] = i;

    State state{std::to_string(i), ary};

    // Ensure we pass an already locked mutex to the cppcb!
    LOG("acquire lock");
    boost::unique_lock<boost::mutex> lock(this->m);

    LOG("call this->pycb(state)");
    this->pycb(state);

    LOG("release Python GIL");
    py::gil_scoped_release release;

    // We've set some events in motion:
    //  - in python, env = Env() has now returned, storing our lambda_act
    //  - in python, env.step(action) will be called, which will call our lambda_act
    // our lambda_act will then call AI->cb->makeAction()
    // ...we wait until that happens, and FINALLY we can return from yourTurn
    LOG("cond.wait()");
    this->cond.wait(lock);

    LOGSTR("this->action.getA()=", this->action.getA());
    LOG("return");
}
