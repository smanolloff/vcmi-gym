#include <cstdio>
// #include <boost/thread.hpp>
#include <pybind11/functional.h>
// #include "myclient.h"
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

// #include <Python.h>
#include <pybind11/pybind11.h>

// #include <boost/asio.hpp>

// WIP version of a connector that modifies a serializable array
// (one initialized in python via multiprocessing.Array(ctypes.c_int, 5))

void start_vcmi(const std::function<void(int[])> &pycallback, int values[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[CPP] (start_vcmi) called\n");
    printf("[CPP] (start_vcmi) values[%d, %d, %d, %d, %d]\n", values[0], values[1], values[2], values[3], values[4]);

    boost::mutex m;
    boost::condition_variable cond;

    // Acquire the lock first to prevent it from locking first
    // (ie. pass an already locked mutex to the lambda)
    printf("[CPP] (start_vcmi) acquire lock\n");
    boost::unique_lock<boost::mutex> lock1(m);


    // <DEBUG>
    // this shows the current thread (same as py thread)
    // already HAS a state, and already HAS a GIL
    // (requested exactly once, by python itself)
    /*
    auto tstate = PyGILState_GetThisThreadState();
    tstate
        ? printf("[CPP] (start_vcmi) tstate->gilstate_counter: %d\n", tstate->gilstate_counter)
        : printf("[CPP] (start_vcmi) tstate NULL\n");
    */
    // </DEBUG>

    // <DEBUG>
    // this shows how acquiring the GIL simply increments
    // the GIL counter, without doing anything else
    // See https://github.com/python/cpython/blob/v3.10.13/Python/pystate.c#L1535
    /*
    printf("[CPP] (start_vcmi) acquire Python GIL\n");
    auto gstate = PyGILState_Ensure();
    tstate = PyGILState_GetThisThreadState();
    tstate
        ? printf("[CPP] (start_vcmi) tstate->gilstate_counter: %d\n", tstate->gilstate_counter)
        : printf("[CPP] (start_vcmi) tstate NULL\n");
    */
    // </DEBUG>

    // GIL management: attempt #1: use core CPython functions
    // https://docs.python.org/3/c-api/init.html#releasing-the-gil-from-extension-code
    // Save the thread state in a local variable.
    // Release the global interpreter lock.
    //
    // PROBLEM: cpp compiler error for 'static' storage class
    /*
    printf("[CPP] (start_vcmi) release Python GIL\n");
    Py_BEGIN_ALLOW_THREADS
    */


    // GIL management: attempt #2: use pybind11 functions
    // https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
    // GIL is held when called from Python code. Release GIL before
    // calling into (potentially long-running) C++ code
    printf("[CPP] (start_vcmi) release Python GIL\n");
    pybind11::gil_scoped_release release;

    printf("[CPP] (start_vcmi) boost::thread(lambda)\n");
    boost::thread([&m, &cond, &pycallback, &values](){
        printf("[CPP] (lambda) called\n");
        printf("[CPP] (lambda) values[%d, %d, %d, %d, %d]\n", values[0], values[1], values[2], values[3], values[4]);

        printf("[CPP] (lambda) acquire lock\n");
        boost::lock_guard<boost::mutex> lock2(m);

        // printf("[CPP] (lambda) sleep 1s\n");
        // boost::this_thread::sleep_for(boost::chrono::seconds(1));

        // GIL management: attempt #1 (cont)
        // https://docs.python.org/3/c-api/init.html#non-python-created-threads
        /*
        printf("[CPP] (lambda) acquire Python GIL\n");
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        */

        // GIL management: attempt #2 (cont)
        // Acquire the GIL
        printf("[CPP] (lambda) acquire Python GIL\n");
        pybind11::gil_scoped_acquire acquire;

        // THIS segfaults!
        // printf("[CPP] (lambda) set values[3]=69\n");
        // values[4] = 69;

        printf("[CPP] (lambda) call pycallback(values)\n");
        pycallback(values);

        // GIL management: attempt #2 (cont)
        printf("[CPP] (lambda) release Python GIL\n");
        pybind11::gil_scoped_release release;

        // GIL management: attempt #1 (cont)
        // https://docs.python.org/3/c-api/init.html#non-python-created-threads
        // Release the thread. No Python API allowed beyond this point.
        /*
        printf("[CPP] (lambda) release Python GIL\n");
        PyGILState_Release(gstate);
        */

        printf("[CPP] (lambda) cond.notify_one()\n");
        cond.notify_one();

        printf("[CPP] (lambda) return\n");
    });

    printf("[CPP] (start_vcmi) sleep 1s\n");
    boost::this_thread::sleep_for(boost::chrono::seconds(1));

    printf("[CPP] (start_vcmi) cond.wait()\n");
    cond.wait(lock1);

    // GIL management: attempt #1 (cont)
    // https://docs.python.org/3/c-api/init.html#releasing-the-gil-from-extension-code
    // Reacquire the global interpreter lock.
    // Restore the thread state from the local variable.
    /*
    printf("[CPP] (start_vcmi) Reacquire Python GIL\n");
    Py_BEGIN_ALLOW_THREADS
    */

    // GIL management: attempt #2 (cont)
    // XXX: the pybind11 examples do not re-acquire the GIL at all?!?
    printf("[CPP] (start_vcmi) Reacquire Python GIL\n");
    pybind11::gil_scoped_acquire acquire;

    printf("[CPP] (start_vcmi) return\n");
}

PYBIND11_MODULE(connector, m) {
    m.def("start_vcmi", &start_vcmi, "Start VCMI");
}
