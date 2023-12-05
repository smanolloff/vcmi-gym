#include <cstdio>

// https://pybind11.readthedocs.io/en/stable/advanced/embedding.html
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "connload.h"
#include "conncommon.h"

namespace py = pybind11;
using namespace pybind11::literals;

// start the interpreter and keep it alive forever
auto guard = new py::scoped_interpreter();
auto scope = py::module_::import("__main__").attr("__dict__");

py::object* model;

MMAI::Export::Action getAction(MMAI::Export::Result* r) {
    if (!model) {
        // init
        printf("Initializing model...\n");
        auto path = "/Users/simo/Projects/vcmi-gym/data/M8-PBT-MPPO-20231204_191243/576e9_00000/checkpoint_000139/model.zip";
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(1, "/Users/simo/Projects/vcmi-gym/.venv/lib/python3.10/site-packages");
        py::eval_file("/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/connector/connload.py", scope);
        auto model_cls = py::object(py::eval("MPPO_AI", scope).cast<py::object>());
        model = new py::object(model_cls(path).cast<py::object>());
    }

    auto ps = P_State(r->state.size());
    auto psmd = ps.mutable_data();
    for (int i=0; i < r->state.size(); i++)
        psmd[i] = r->state[i].norm;

    auto pam = P_ActMask(r->actmask.size());
    auto pammd = pam.mutable_data();
    for (int i=0; i < r->actmask.size(); i++)
        pammd[i] = r->actmask[i];

    auto predict = model->attr("predict");
    auto result = predict(ps, pam).cast<MMAI::Export::Action>();

    printf("PREDICTION: %d\n", result);

    return result;
}
