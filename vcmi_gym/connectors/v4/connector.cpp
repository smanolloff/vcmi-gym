#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

#include "procconnector.h"
#include "threadconnector.h"

namespace Connector::V4 {
    namespace py = pybind11;

    PYBIND11_MODULE(connector_v4, m) {
        py::class_<P_State>(m, "P_State")
            .def("get_state", &P_State::get_state)
            .def("get_actmask", &P_State::get_actmask)
            .def("get_attnmask", &P_State::get_attnmask)
            .def("get_errcode", &P_State::get_errcode)
            .def("get_side", &P_State::get_side)
            .def("get_dmg_dealt", &P_State::get_dmg_dealt)
            .def("get_dmg_received", &P_State::get_dmg_received)
            .def("get_units_lost", &P_State::get_units_lost)
            .def("get_units_killed", &P_State::get_units_killed)
            .def("get_value_lost", &P_State::get_value_lost)
            .def("get_value_killed", &P_State::get_value_killed)
            .def("get_initial_side0_army_value", &P_State::get_initial_side0_army_value)
            .def("get_initial_side1_army_value", &P_State::get_initial_side1_army_value)
            .def("get_current_side0_army_value", &P_State::get_current_side0_army_value)
            .def("get_current_side1_army_value", &P_State::get_current_side1_army_value)
            .def("get_is_battle_over", &P_State::get_is_battle_over)
            .def("get_is_victorious", &P_State::get_is_victorious);

        py::class_<Proc::Connector, std::unique_ptr<Proc::Connector>>(m, "ProcConnector")
            .def(py::init<
                const int &,         // maxlogs
                const std::string &, // mapname
                const int &,         // seed
                const int &,         // randomHeroes
                const int &,         // randomObstacles
                const int &,         // townChance
                const int &,         // warmachineChance
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
            .def("start", &Thread::Connector::start)
            .def("reset", &Thread::Connector::reset)
            .def("connect", &Thread::Connector::connect)
            .def("step", &Thread::Connector::step)
            .def("render", &Thread::Connector::render)
            .def("getLogs", &Thread::Connector::getLogs);

        py::register_exception<Thread::VCMIConnectorException>(m, "PyThreadVCMIConnectorException");

    }

}
