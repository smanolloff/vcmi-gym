#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

#include "procconnector.h"
#include "threadconnector.h"

namespace Connector::V9 {
    namespace py = pybind11;

    PYBIND11_MODULE(connector_v9, m) {
        py::class_<P_State>(m, "P_State")
            .def("get_state", &P_State::get_state)
            .def("get_actmask", &P_State::get_actmask)
            // .def("get_edge_sources", &P_State::get_edge_sources)
            // .def("get_edge_targets", &P_State::get_edge_targets)
            // .def("get_edge_values", &P_State::get_edge_values)
            // .def("get_edge_types", &P_State::get_edge_types)
            .def("get_edge_attrs", &P_State::get_edge_attrs)
            .def("get_edge_index", &P_State::get_edge_index)
            .def("get_errcode", &P_State::get_errcode);

        py::class_<Proc::Connector, std::unique_ptr<Proc::Connector>>(m, "ProcConnector")
            .def(py::init<
                const int &,         // maxlogs
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
            .def("start", &Proc::Connector::start)
            .def("reset", &Proc::Connector::reset)
            .def("step", &Proc::Connector::step)
            .def("render", &Proc::Connector::render)
            .def("getLogs", &Proc::Connector::getLogs);

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
