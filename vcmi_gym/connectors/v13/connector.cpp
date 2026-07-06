#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

#include "threadconnector.h"

namespace Connector::V13 {
    namespace py = pybind11;

    PYBIND11_MODULE(connector_v13, m) {
        py::class_<P_State>(m, "P_State")
            .def("get_state", &P_State::get_state)
            .def("get_action_mask", &P_State::get_action_mask)
            .def("get_links_dict", &P_State::get_links_dict)
            .def("get_errcode", &P_State::get_errcode);

        py::class_<Thread::Connector, std::unique_ptr<Thread::Connector>>(m, "ThreadConnector")
            .def(py::init<
                int,                    // maxlogs
                int,                    // bootTimeout
                int,                    // vcmiTimeout
                int,                    // userTimeout
                const std::string &,    // red
                const std::string &,    // redModel
                const std::string &,    // blue
                const std::string &,    // blueModel

                const std::string &,    // mapname
                int,                    // seed
                int,                    // randomHeroes
                int,                    // randomObstacles
                int,                    // townChance
                int,                    // warmachineChance
                bool,                   // randomArmies
                int,                    // randomArmyValueMin
                int,                    // randomArmyValueMax
                int,                    // randomArmyTargetVar
                int,                    // tightFormationChance
                int,                    // randomTerrainChance
                int,                    // leftVipChance
                int,                    // rightVipChance
                const std::string &,    // battlefieldPattern
                int,                    // manaMin
                int,                    // manaMax
                int,                    // randomPrimarySkills
                int,                    // swapSides
                const std::string &,    // loglevelGlobal
                const std::string &,    // loglevelAI
                const std::string &,    // loglevelNetwork
                const std::string &,    // loglevelStats
                bool,                   // redAllowMlBot
                bool,                   // blueAllowMlBot
                const std::string &,    // statsMode
                const std::string &,    // statsStorage
                int                     // statsPersistFreq
            >(),
                py::arg("maxlogs"),
                py::arg("bootTimeout"),
                py::arg("vcmiTimeout"),
                py::arg("userTimeout"),
                py::arg("red"),
                py::arg("redModel"),
                py::arg("blue"),
                py::arg("blueModel"),
                py::arg("mapname"),
                py::arg("seed"),
                py::arg("randomHeroes"),
                py::arg("randomObstacles"),
                py::arg("townChance"),
                py::arg("warmachineChance"),
                py::arg("randomArmies"),
                py::arg("randomArmyValueMin"),
                py::arg("randomArmyValueMax"),
                py::arg("randomArmyTargetVar"),
                py::arg("tightFormationChance"),
                py::arg("randomTerrainChance"),
                py::arg("leftVipChance"),
                py::arg("rightVipChance"),
                py::arg("battlefieldPattern"),
                py::arg("manaMin"),
                py::arg("manaMax"),
                py::arg("randomPrimarySkills"),
                py::arg("swapSides"),
                py::arg("loglevelGlobal"),
                py::arg("loglevelAI"),
                py::arg("loglevelNetwork"),
                py::arg("loglevelStats"),
                py::arg("redAllowMlBot"),
                py::arg("blueAllowMlBot"),
                py::arg("statsMode"),
                py::arg("statsStorage"),
                py::arg("statsPersistFreq")
            )
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
