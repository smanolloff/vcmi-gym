#include <cassert>
#include <cstdio>

// https://pybind11.readthedocs.io/en/stable/advanced/embedding.html
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "loader.h"
#include "conncommon.h"

namespace py = pybind11;
using namespace pybind11::literals;

// start the interpreter and keep it alive until shutdown
// Apparently, loading the shared lib multiple times will not duplicate
// any resources, so global vars remain
// https://stackoverflow.com/a/22863121
auto guard = new py::scoped_interpreter();

py::object* model;
PyThreadState *_save;

// low-level acquire/release
#define GIL_LOW_ACQUIRE() { LOG("ACQUIRING GIL..."); assert(_save); PyEval_RestoreThread(_save); _save = nullptr; LOG("ACQUIRED GIL."); }
#define GIL_LOW_RELEASE() { LOG("RELEASING GIL..."); assert(!_save); _save = PyEval_SaveThread(); LOG("RELEASED GIL."); }

#define GIL_ACQUIRE() LOG("ACQUIRING GIL..."); auto gstate = PyGILState_Ensure(); LOG("ACQUIRED GIL.");
#define GIL_RELEASE() LOG("RELEASING GIL..."); PyGILState_Release(gstate); LOG("ACQUIRED GIL.");

void ConnectorLoader_init(std::string path) {
    LOG("start");

    assert(PyGILState_Check());

    LOGSTR("Loading model from ", path);

    {
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(1, "/Users/simo/Projects/vcmi-gym/.venv/lib/python3.10/site-packages");
        py::eval_file("/Users/simo/Projects/vcmi-gym/vcmi_gym/envs/v0/connector/loader.py");
        auto model_cls = py::object(py::eval("Loader.MPPO").cast<py::object>());
        model = new py::object(model_cls(path).cast<py::object>());
    }

    // the only way to do a non-scoped GIL release is directly via PyEval_SaveThread()
    GIL_LOW_RELEASE();
    LOG("return");
}

MMAI::Export::Action ConnectorLoader_getAction(const MMAI::Export::Result* r) {
    LOG("start");

    if (!guard) {
        throw std::runtime_error("Already shutdown");
    }

    MMAI::Export::Action result;

    {
        // using low-level PyEval_RestoreThread for acquiring GIL here is not an option
        // (getAction is called from another thread => PyEval_RestoreThread is a noop)
        // using higher-level PyGILState_Ensure + PyGILState_Release works (I think)
        // ...but py::gil_scoped_acquire also works and is the simplest approach
        py::gil_scoped_acquire acquire;
        auto ps = P_State(r->state.size());
        auto psmd = ps.mutable_data();
        for (int i=0; i < r->state.size(); i++)
            psmd[i] = r->state[i].norm;

        LOG("inbetween")

        auto pam = P_ActMask(r->actmask.size());
        auto pammd = pam.mutable_data();
        for (int i=0; i < r->actmask.size(); i++)
            pammd[i] = r->actmask[i];

        auto predict = model->attr("predict");
        result = predict(ps, pam).cast<MMAI::Export::Action>();
    }

    LOGSTR("return ", std::to_string(result));
    return result;
}