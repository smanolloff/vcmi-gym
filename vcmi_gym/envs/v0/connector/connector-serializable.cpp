#include <cstdio>
// #include <boost/thread.hpp>
#include <pybind11/functional.h>
// #include "myclient.h"
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <pthread.h>

// #include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// #include <boost/asio.hpp>

namespace py = pybind11;

struct Action {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }
};

struct State {
    void setA(const std::string &a_) { a = a_; }
    void setB(const py::array_t<float> &b_) { b = b_; }

    const std::string &getA() const { return a; }
    const py::array_t<float> &getB() const { return b; }
};

// this can't be defined here as const, as it will need access to
// a cond variable which it should modify
// TODO: what if the cond variable is global?
// const State act(Action a) {
//     printf("[CPP] (act) called with action.getA(): %s, action.getB(): ?\n", a.getA());
// }

void start_vcmi(const std::function<Action(State)> &pycallback) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[CPP] (start_vcmi) called\n");
    pthread_setname_np("VCMIRoot");

    // https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
    // GIL is held when called from Python code. Release GIL before
    // calling into (potentially long-running) C++ code
    printf("[CPP] (start_vcmi) release Python GIL\n");
    py::gil_scoped_release release;

    printf("[CPP] (start_vcmi) boost::thread(lambda_yourTurn)\n");
    boost::thread([&pycallback]() -> const std::function<State(Action)> {
        printf("[CPP] (lambda_yourTurn) called\n");
        printf("[CPP] (lambda_yourTurn) values[%d, %d, %d, %d, %d]\n", values[0], values[1], values[2], values[3], values[4]);

        pthread_setname_np("VCMIClient");

        printf("[CPP] (lambda_yourTurn) acquire lock\n");
        boost::lock_guard<boost::mutex> lock2(m);

        // printf("[CPP] (lambda_yourTurn) sleep 1s\n");
        // boost::this_thread::sleep_for(boost::chrono::seconds(1));

        boost::mutex m;
        boost::condition_variable cond;

        // Define a callback to be called after in env.step()
        // TODO: lambda_act
        auto [&m, &cond]

        // Acquire the lock now to prevent the lambda_act from locking first
        // (ie. pass an already locked mutex to the lambda_act)
        printf("[CPP] (lambda_yourTurn) acquire lock\n");
        boost::unique_lock<boost::mutex> lock(m);

        // Acquire the GIL
        printf("[CPP] (lambda_yourTurn) acquire Python GIL\n");
        py::gil_scoped_acquire acquire;

        // THIS segfaults!
        // printf("[CPP] (lambda_yourTurn) set values[3]=69\n");
        // values[4] = 69;

        printf("[CPP] (lambda_yourTurn) call pycallback(values)\n");
        pycallback(values);

        printf("[CPP] (lambda_yourTurn) release Python GIL\n");
        py::gil_scoped_release release;

        // We've set some events in motion:
        //  - in python, env = Env() has now returned, storing our lambda_act
        //  - in python, env.step(action) will be called, which will call our lambda_act
        // our lambda_act will then call AI->cb->makeAction()
        // ...we wait until that happens, and FINALLY we can return from yourTurn
        printf("[CPP] (lambda_yourTurn) cond.wait()\n");
        cond.wait(lock)

        printf("[CPP] (lambda_yourTurn) return\n");
    });

    // printf("[CPP] (start_vcmi) Reacquire Python GIL\n");
    // py::gil_scoped_acquire acquire;

    printf("[CPP] (start_vcmi) Entering sleep loop...\n");
    while(true)
        boost::this_thread::sleep(boost::chrono::seconds(1));

    printf("[CPP] (start_vcmi) return !!!! !SHOULD NEVER HAPPEN\n");
}

PYBIND11_MODULE(connector, m) {
    m.def("start_vcmi", &start_vcmi, "Start VCMI");

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def("setA", &Pet::setA)
        .def("getA", &Pet::getA)
        .def("setB", &Pet::setB)
        .def("getB", &Pet::getB);

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def("setA", &Pet::setA)
        .def("getA", &Pet::getA)
        .def("setB", &Pet::setB)
        .def("getB", &Pet::getB);
}
