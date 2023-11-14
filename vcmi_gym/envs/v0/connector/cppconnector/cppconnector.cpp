#include <memory>
#include <pybind11/pybind11.h>
#include "cppconnector.h"
#include "aitypes.h"
#include "pyclient.h" // "vendor" header file

int i = 0;

#define REPORT_GIL() LOGSTR("GIL: ", std::to_string(PyGILState_Check()))

const auto cc = std::make_unique<CppConnector>("pikemen.vmap", "trace");

void prestart() {
    cc->prestart();
}

void start() {
    cc->start();
}

void act(MMAI::Action a) {
    cc->act(a);
}

std::unique_ptr<CppConnector> create_cppconnector(const std::string mapname_, const std::string loglevel_) {
    // cc = std::make_unique<CppConnector>(mapname_, loglevel_);
    return std::make_unique<CppConnector>(mapname_, loglevel_);
}

CppConnector::~CppConnector() {
    // not the issue - it seems it's not called
    LOG("---------------------------- DESTRUCTED ---------------------");
}

CppConnector::CppConnector(const std::string mapname_, const std::string loglevel_)
: mapname(mapname_), loglevel(loglevel_) {
    LOG("++++ CppConnector CONSTRUCTOR ++++")
    auto getaction = [this](MMAI::Result r) {
        LOG("GETACTIONNNNNNNNNNNN");
        return this->getActionDummy(r);
    };
    cbprovider->f_getAction = getaction;
    cbprovider->debugstr = "reassigned debugstr in CppConnector constructor";
}

const MMAI::Action CppConnector::getActionDummy(MMAI::Result r) {
    return MMAI::Action(3);
}

void CppConnector::act(MMAI::Action a) {
    py::gil_scoped_release release;
    LOG("sleep 30s...");
    std::this_thread::sleep_for(std::chrono::seconds(30));
}

// THIS returns => we have issue
// move to start() for debugging..
void CppConnector::prestart() {
    // init_vcmi(loglevel, cbprovider);
    // LOG("return");
}

void CppConnector::start() {
    setvbuf(stdout, NULL, _IONBF, 0);
    LOG("start");

    init_vcmi(loglevel, cbprovider.get());

    vcmithread = std::thread([]() {
        LOG("[thread] Start VCMI");
        start_vcmi("pikemen.vmap");
        assert(false); // should never happen
    });

    py::gil_scoped_release release;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    LOG("===== START RETURN ======")
}

void CppConnector::renderAnsi() {};
void CppConnector::reset() {};

static const int get_state_size() { return MMAI::STATE_SIZE; }
static const int get_n_actions() { return MMAI::N_ACTIONS; }
static const std::map<MMAI::ErrMask, std::tuple<std::string, std::string>> get_error_mapping() {
  std::map<MMAI::ErrMask, std::tuple<std::string, std::string>> res;

  for (auto& [_err, tuple] : MMAI::ERRORS) {
    res[std::get<0>(tuple)] = {std::get<1>(tuple), std::get<2>(tuple)};
  }

  return res;
}

PYBIND11_MODULE(cppconnector, m) {
    m.def("get_state_size", &get_state_size, "Get number of elements in state");
    m.def("get_n_actions", &get_n_actions, "Get max expected value of action");
    m.def("get_error_mapping", &get_error_mapping, "Get available error names and flags");
    m.def("create_cppconnector", &create_cppconnector, "");

    m.def("prestart", &prestart);
    m.def("start", &start);
    m.def("act", &act);

    py::class_<P_Result>(m, "P_Result")
        .def("get_state", &P_Result::get_state)
        .def("get_errmask", &P_Result::get_errmask)
        .def("get_dmg_dealt", &P_Result::get_dmg_dealt)
        .def("get_dmg_received", &P_Result::get_dmg_received)
        .def("get_units_lost", &P_Result::get_units_lost)
        .def("get_units_killed", &P_Result::get_units_killed)
        .def("get_value_lost", &P_Result::get_value_lost)
        .def("get_value_killed", &P_Result::get_value_killed)
        .def("get_is_battle_over", &P_Result::get_is_battle_over)
        .def("get_is_victorious", &P_Result::get_is_victorious);

    py::class_<CppConnector>(m, "CppConnector")
        .def(py::init<const std::string &, const std::string &>())
        .def("prestart", &CppConnector::prestart)
        .def("start", &CppConnector::start)
        .def("reset", &CppConnector::reset)
        .def("act", &CppConnector::act)
        .def("renderAnsi", &CppConnector::renderAnsi);
}
