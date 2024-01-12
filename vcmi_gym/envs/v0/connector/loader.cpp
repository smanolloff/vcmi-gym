// =============================================================================
// Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <cassert>
#include <cstdio>

// https://pybind11.readthedocs.io/en/stable/advanced/embedding.html
#include <filesystem>
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "loader.h"
#include "conncommon.h"
#include "mmai_export.h"

namespace py = pybind11;
using namespace pybind11::literals;

py::scoped_interpreter* guard;

std::map<MMAI::Export::Side, py::object*> models;
PyThreadState* _save;

// low-level acquire/release
#define GIL_LOW_ACQUIRE() { LOG("ACQUIRING GIL..."); assert(_save); PyEval_RestoreThread(_save); _save = nullptr; LOG("ACQUIRED GIL."); }
#define GIL_LOW_RELEASE() { LOG("RELEASING GIL..."); assert(!_save); _save = PyEval_SaveThread(); LOG("RELEASED GIL."); }

#define GIL_ACQUIRE() LOG("ACQUIRING GIL..."); auto gstate = PyGILState_Ensure(); LOG("ACQUIRED GIL.");
#define GIL_RELEASE() LOG("RELEASING GIL..."); PyGILState_Release(gstate); LOG("ACQUIRED GIL.");

void init(MMAI::Export::Side side, std::string gymdir, std::string modelpath) {
    LOG("start");

    // This ptr will call the destructor once it goes out of scope
    std::unique_ptr<py::gil_scoped_acquire> acquire;

    auto wasInitialized = Py_IsInitialized();

    if (wasInitialized) {
        LOG("*** Interpreter already initialized ***");
        // Interpreter running - means PID started as Python code
        // Just acquire the GIL in this case
        acquire = std::make_unique<py::gil_scoped_acquire>();
    } else {
        // Interpreter not running - means PID started as C++ code
        // start the interpreter and keep it alive until shutdown
        // (it will automatically acquire the GIL)
        LOG("!!! Starting embedded Python interpreter !!!");
        guard = new py::scoped_interpreter();
    }

    // at this point, there must be a python interpreter
    assert(PyGILState_Check());

    LOG("Loading model " + std::to_string(static_cast<int>(side)) + " from " + modelpath);

    {
        auto oldwd = std::filesystem::current_path();
        std::filesystem::current_path(gymdir);
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(1, ".venv/lib/python3.10/site-packages");
        sys.attr("path").attr("insert")(1, "vcmi_gym/envs/v0/connector");
        py::eval_file(("vcmi_gym/envs/v0/connector/loader.py"));
        auto model_cls = py::object(py::eval("Loader.MPPO").cast<py::object>());
        assert(models.count(side) == 0);
        models[side] = new py::object(model_cls(modelpath).cast<py::object>());
        std::filesystem::current_path(oldwd);
    }

    // If the interpreter was just started, it automatically acquired the
    // the only way to do a non-scoped GIL release is directly via PyEval_SaveThread()
    if (!wasInitialized)
        GIL_LOW_RELEASE();

    LOG("return");
}

void ConnectorLoader_initAttacker(std::string gymdir, std::string modelpath) {
    auto side = MMAI::Export::Side::ATTACKER;
    LOG("Initializing model for ATTACKER (#" + std::to_string(static_cast<int>(side)) + ")");
    init(side, gymdir, modelpath);
}

void ConnectorLoader_initDefender(std::string gymdir, std::string modelpath) {
    auto side = MMAI::Export::Side::DEFENDER;
    LOG("Initializing model for DEFENDER (#" + std::to_string(static_cast<int>(side)) + ")");
    init(side, gymdir, modelpath);
}

MMAI::Export::Action getAction(MMAI::Export::Side side, const MMAI::Export::Result* &r) {
    LOG("start");

    if (!Py_IsInitialized()) {
        throw std::runtime_error("Not initialized or already shutdown");
    }

    MMAI::Export::Action result;

    {
        // using low-level PyEval_RestoreThread for acquiring GIL here is not an option
        // (getAction is called from another thread => PyEval_RestoreThread is a noop)
        // using higher-level PyGILState_Ensure + PyGILState_Release works (I think)
        // ...but py::gil_scoped_acquire also works and is the simplest approach
        py::gil_scoped_acquire acquire;
        assert(models.count(side) == 1);
        auto ps = P_State(r->state.size());
        auto psmd = ps.mutable_data();
        for (int i=0; i < r->state.size(); i++) {
            if (r->state[i].norm > 1 || r->state[i].norm < 0) {
                printf("Bad state[%d]: %f\n", i, r->state[i].norm);
            }

            psmd[i] = r->state[i].norm;
        }

        auto pam = P_ActMask(r->actmask.size());
        auto pammd = pam.mutable_data();
        for (int i=0; i < r->actmask.size(); i++)
            pammd[i] = r->actmask[i];

        if (r->ended) {
            result = MMAI::Export::ACTION_RESET;
        } else {
            auto predict = models[side]->attr("predict");
            result = predict(ps, pam).cast<MMAI::Export::Action>();
        }
    }

    LOG("return action for #" + std::to_string(static_cast<int>(side)) + ": " + std::to_string(result));
    return result;
}


MMAI::Export::Action ConnectorLoader_getActionDefender(const MMAI::Export::Result* r) {
    return getAction(MMAI::Export::Side::DEFENDER, r);
}

MMAI::Export::Action ConnectorLoader_getActionAttacker(const MMAI::Export::Result* r) {
    return getAction(MMAI::Export::Side::ATTACKER, r);
}
