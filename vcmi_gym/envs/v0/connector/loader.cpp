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

// TODO: namespace global vars
py::scoped_interpreter* GUARD;
std::string ENCODING;
std::map<MMAI::Export::Side, py::object*> MODELS;
PyThreadState* _SAVE;

// low-level acquire/release
#define GIL_LOW_ACQUIRE() { LOG("ACQUIRING GIL..."); assert(_SAVE); PyEval_RestoreThread(_SAVE); _SAVE = nullptr; LOG("ACQUIRED GIL."); }
#define GIL_LOW_RELEASE() { LOG("RELEASING GIL..."); assert(!_SAVE); _SAVE = PyEval_SaveThread(); LOG("RELEASED GIL."); }

#define GIL_ACQUIRE() LOG("ACQUIRING GIL..."); auto gstate = PyGILState_Ensure(); LOG("ACQUIRED GIL.");
#define GIL_RELEASE() LOG("RELEASING GIL..."); PyGILState_Release(gstate); LOG("ACQUIRED GIL.");

void init(std::string encoding, MMAI::Export::Side side, std::string gymdir, std::string modelpath) {
    LOG("start");

    if (!(encoding == MMAI::Export::STATE_ENCODING_DEFAULT || encoding == MMAI::Export::STATE_ENCODING_FLOAT))
        throw std::runtime_error("Loader received an invalid encoding: " + encoding);

    ENCODING = encoding;

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
        GUARD = new py::scoped_interpreter();
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
        assert(MODELS.count(side) == 0);
        MODELS[side] = new py::object(model_cls(modelpath).cast<py::object>());
        std::filesystem::current_path(oldwd);
    }

    // If the interpreter was just started, it automatically acquired the
    // the only way to do a non-scoped GIL release is directly via PyEval_SaveThread()
    if (!wasInitialized)
        GIL_LOW_RELEASE();

    LOG("return");
}

void ConnectorLoader_initAttacker(std::string encoding, std::string gymdir, std::string modelpath) {
    auto side = MMAI::Export::Side::LEFT;
    LOG("Initializing model for LEFT (#" + std::to_string(static_cast<int>(side)) + ")");
    init(encoding, side, gymdir, modelpath);
}

void ConnectorLoader_initDefender(std::string encoding, std::string gymdir, std::string modelpath) {
    auto side = MMAI::Export::Side::RIGHT;
    LOG("Initializing model for RIGHT (#" + std::to_string(static_cast<int>(side)) + ")");
    init(encoding, side, gymdir, modelpath);
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
        assert(MODELS.count(side) == 1);

        P_State ps;

        if (ENCODING == MMAI::Export::STATE_ENCODING_DEFAULT) {
            auto vec = MMAI::Export::State{};
            vec.reserve(MMAI::Export::STATE_SIZE_DEFAULT);

            for (auto &u : r->stateUnencoded)
                u.encode(vec);

            if (vec.size() != MMAI::Export::STATE_SIZE_DEFAULT)
                throw std::runtime_error("Unexpected state size: " + std::to_string(vec.size()));

            ps = P_State(MMAI::Export::STATE_SIZE_DEFAULT);
            auto psmd = ps.mutable_data();

            for (int i=0; i<MMAI::Export::STATE_SIZE_DEFAULT; i++)
                psmd[i] = vec[i];

        } else if (ENCODING == MMAI::Export::STATE_ENCODING_FLOAT) {
            if (r->stateUnencoded.size() != MMAI::Export::STATE_SIZE_FLOAT)
                throw std::runtime_error("Unexpected state size: " + std::to_string(r->stateUnencoded.size()));

            ps = P_State(MMAI::Export::STATE_SIZE_FLOAT);
            auto psmd = ps.mutable_data();

            for (int i=0; i<MMAI::Export::STATE_SIZE_FLOAT; i++)
                psmd[i] = r->stateUnencoded[i].encode2Floating();
        } else {
            throw std::runtime_error("Unexpected encoding: " + ENCODING);
        };

        auto pam = P_ActMask(r->actmask.size());
        auto pammd = pam.mutable_data();
        for (int i=0; i < r->actmask.size(); i++)
            pammd[i] = r->actmask[i];

        if (r->ended) {
            result = MMAI::Export::ACTION_RESET;
        } else {
            auto predict = MODELS[side]->attr("predict");
            result = predict(ps, pam).cast<MMAI::Export::Action>();
        }
    }

    LOG("return action for #" + std::to_string(static_cast<int>(side)) + ": " + std::to_string(result));
    return result;
}


MMAI::Export::Action ConnectorLoader_getActionDefender(const MMAI::Export::Result* r) {
    return getAction(MMAI::Export::Side::RIGHT, r);
}

MMAI::Export::Action ConnectorLoader_getActionAttacker(const MMAI::Export::Result* r) {
    return getAction(MMAI::Export::Side::LEFT, r);
}
