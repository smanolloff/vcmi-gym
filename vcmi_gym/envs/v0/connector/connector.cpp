#include <cstdio>
#include <pybind11/pybind11.h>
// #include <myclient.h>

void cpp_function(const std::function<void()>& callback) {
    printf(">>> in connector.cpp ...\n");

    // Perform some C++ work
    // Call the Python callback function
    callback();
}

PYBIND11_MODULE(vcmi_connector, m) {
    m.def("cpp_function", &cpp_function, "Call C++ function with a Python callback");
}
