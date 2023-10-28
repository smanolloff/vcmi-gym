#include <cstdio>
// #include <boost/thread.hpp>
#include <pybind11/functional.h>
// #include "myclient.h"
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <pthread.h>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// #include <boost/asio.hpp>

namespace py = pybind11;

struct Action {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }

    std::string a;
    py::array_t<float> b;
};

struct State {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }

    std::string a;
    py::array_t<float> b;
};

// couldn't figure out a way to have CppCB accept PyCB (declaration recursion)
using CppCB = std::function<void(Action)>;
using PyCB = std::function<void(State)>;
using PyCBInit = std::function<void(CppCB)>;


// this can't be defined here as const, as it will need access to
// a cond variable which it should modify
// TODO: what if the cond variable is global?
// const State act(Action a) {
//     printf("[CPP] (act) called with action.getA(): %s, action.getB(): ?\n", action.getA());
// }

void start_vcmi(const PyCB &pycb, const PyCBInit &pycbinit) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[CPP] (start_vcmi) called\n");

    bool inited = false;

    // XXX: pycb is &bound (or stored as instance var in AI)
    //      and reused in subsequent calls.
    //      it means it must never be GCd in python -- eg. by
    //      making PyConnector a class and storing the CB in it
    //      (gym env should instantiate PyConnector once at __init__)
    printf("[CPP] (start_vcmi) boost::thread(vcmiclient)\n");
    boost::thread([&pycb, &pycbinit, &inited]() {
        printf("[CPP] (vcmiclient) called\n");

        // printf("[CPP] (vcmiclient) sleep 1s\n");
        // boost::this_thread::sleep_for(boost::chrono::seconds(1));

        boost::mutex m;
        boost::condition_variable cond;
        // Acquire the GIL
        printf("[CPP] (vcmiclient) acquire Python GIL\n");
        py::gil_scoped_acquire acquire;
        Action my_action{};

        // Define a callback to be called after in env.step()
        auto cppcb = [&m, &cond, &my_action](Action action) {
            printf("[CPP] (cppcb) called with action.getA()=%s\n", action.getA().c_str());

            // TODO: no need to acquire GIL here?
            // (caller is a Python thread)
            printf("[CPP] (cppcb) acquire lock");
            boost::lock_guard<boost::mutex> lock(m);

            printf("[CPP] (cppcb) assign my_action = action");
            my_action = action;

            printf("[CPP] (cppcb) cond.notify_one()");
            cond.notify_one();

            printf("[CPP] (cppcb) return");
        };

        // THIS segfaults!
        // Probably because values was declared as int[] instead py_array_t<int>
        // printf("[CPP] (vcmiclient) set values[3]=69\n");
        // values[4] = 69;

        // Simulate state received from vcmi
        printf("[CPP] (vcmiclient) set state={a='[4,5,6]', b=[4,5,6]}\n");
        State state{std::string("[4,5,6]"), py::array_t<float>({ 4, 5, 6})};

        // Acquire the lock now to prevent the lambda_act from locking first
        // (ie. pass an already locked mutex to the lambda_act)
        printf("[CPP] (vcmiclient) acquire lock\n");
        boost::unique_lock<boost::mutex> lock(m);

        // XXX: Danger: SIGSERV?
        // NOTE: this function will modify Python vars: state, cppcb
        //       my AI class. I don't such a class in this simulation
        printf("[CPP] (vcmiclient) DANGER call pycbinit(cppcb)\n");
        pycbinit(cppcb);

        printf("[CPP] (vcmiclient) set inited = true\n");
        inited = true;

        printf("[CPP] (vcmiclient) call pycb(state)\n");
        pycb(state);

        printf("[CPP] (vcmiclient) release Python GIL\n");
        py::gil_scoped_release release;

        // We've set some events in motion:
        //  - in python, env = Env() has now returned, storing our lambda_act
        //  - in python, env.step(action) will be called, which will call our lambda_act
        // our lambda_act will then call AI->cb->makeAction()
        // ...we wait until that happens, and FINALLY we can return from yourTurn
        printf("[CPP] (vcmiclient) cond.wait()\n");
        cond.wait(lock);

        printf("[CPP] (vcmiclient) my_action.getA()=%s\n", my_action.getA().c_str());

        printf("[CPP] (vcmiclient) return\n");
    });


    // https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
    // GIL is held when called from Python code. Release GIL before
    // calling into (potentially long-running) C++ code
    printf("[CPP] (start_vcmi) release Python GIL\n");
    py::gil_scoped_release release;

    printf("[CPP] (start_vcmi) Entering sleep loop...\n");
    while(true)
        boost::this_thread::sleep_for(boost::chrono::seconds(1));

    printf("[CPP] (start_vcmi) return !!!! !SHOULD NEVER HAPPEN\n");
}

PYBIND11_MODULE(connector, m) {
    m.def("start_vcmi", &start_vcmi, "Start VCMI");

    // py::class_<Action>(m, "Action")
    //     .def(py::init<>())
    //     .def("setA", &Action::setA)
    //     .def("getA", &Action::getA)
    //     .def("setB", &Action::setB)
    //     .def("getB", &Action::getB);

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def("setA", &State::setA)
        .def("getA", &State::getA)
        .def("setB", &State::setB)
        .def("getB", &State::getB);
}
