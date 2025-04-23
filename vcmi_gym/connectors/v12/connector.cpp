#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

#include "threadconnector.h"

namespace Connector::V12 {
    namespace py = pybind11;

    PYBIND11_MODULE(connector_v12, m) {
        py::class_<P_State>(m, "P_State")
            .def("get_intermediate_states", &P_State::get_intermediate_states)
            .def("get_intermediate_action_masks", &P_State::get_intermediate_action_masks)
            .def("get_intermediate_actions", &P_State::get_intermediate_actions)
            .def("get_errcode", &P_State::get_errcode);

        py::class_<Thread::Connector, std::unique_ptr<Thread::Connector>>(m, "ThreadConnector")
            .def(py::init<
                const int &,         // maxlogs
                const int &,         // bootTimeout
                const int &,         // vcmiTimeout
                const int &,         // userTimeout
                const std::string &, // mapname
                const int &,         // seed
                const int &,         // randomHeroes
                const int &,         // randomObstacles
                const int &,         // townChance
                const int &,         // warmachineChance
                const int &,         // tightFormationChance
                const int &,         // randomTerrainChance
                const std::string &, // battlefieldPattern
                const int &,         // manaMin
                const int &,         // manaMax
                const int &,         // swapSides
                const std::string &, // loglevelGlobal
                const std::string &, // loglevelAI
                const std::string &, // loglevelStats
                const std::string &, // red
                const std::string &, // blue
                const std::string &, // redModel
                const std::string &, // blueModel
                const std::string &, // statsMode
                const std::string &, // statsStorage
                const int &          // statsPersistFreq
            >())
            .def("shutdown", &Thread::Connector::shutdown)
            .def("start", &Thread::Connector::start)
            .def("reset", &Thread::Connector::reset)
            .def("connect", &Thread::Connector::connect)
            .def("step", &Thread::Connector::step)
            .def("render", &Thread::Connector::render)
            .def("getLogs", &Thread::Connector::getLogs);

        py::register_exception<Thread::VCMIConnectorException>(m, "PyThreadVCMIConnectorException");

    }

}
