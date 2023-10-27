#include <cstdio>
// #include <boost/thread.hpp>
#include <pybind11/functional.h>
// #include "myclient.h"
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
// #include <boost/asio.hpp>

void start_vcmi(const std::function<void(int)> &pycallback, int value) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[CPP] (start_vcmi) called with pycallback=?, value=%d\n", value);

    boost::mutex m;
    boost::condition_variable cond;

    // Acquire the lock first to prevent it from locking first
    // (ie. pass an already locked mutex to the lambda)
    printf("[CPP] (start_vcmi) acquire lock\n");
    boost::unique_lock<boost::mutex> lock1(m);

    printf("[CPP] (start_vcmi) boost::thread(lambda)\n");
    boost::thread([&m, &cond, &pycallback, &value](){
        printf("[CPP] (lambda) called with value=%d\n", value);

        printf("[CPP] (lambda) acquire lock\n");
        boost::lock_guard<boost::mutex> lock2(m);

        printf("[CPP] (lambda) sleep 1s\n");
        boost::this_thread::sleep_for(boost::chrono::seconds(1));

        printf("[CPP] (lambda) call pycallback(value+1)\n");
        pycallback(value+1);

        printf("[CPP] (lambda) cond.notify_one()\n");
        cond.notify_one();

        printf("[CPP] (lambda) return\n");
    });

    printf("[CPP] (start_vcmi) sleep 1s\n");
    boost::this_thread::sleep_for(boost::chrono::seconds(1));

    printf("[CPP] (start_vcmi) cond.wait()\n");
    cond.wait(lock1);

    printf("[CPP] (start_vcmi) return\n");
}

PYBIND11_MODULE(connector, m) {
    m.def("start_vcmi", &start_vcmi, "Start VCMI");
}
