from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain
from conans import tools

required_conan_version = ">=1.51.3"


class Connector(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = ["pybind11/[~2.12.0]", "boost/[^1.69 <1.86]"]
    default_options = {"boost/*:shared": True}

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()

        deps = CMakeDeps(self)
        if tools.get_env("GENERATE_ONLY_BUILT_CONFIG", default=False):
            deps.generate()
            return

        configs = [
            "Debug",
            # "MinSizeRel",
            "Release",
            # "RelWithDebInfo",
        ]

        for config in configs:
            print(f"generating CMakeDeps for {config}")
            deps.configuration = config
            deps.generate()

    def configure(self):
        self.options["boost"].without_context = True
        self.options["boost"].without_contract = True
        self.options["boost"].without_coroutine = True
        self.options["boost"].without_fiber = True
        self.options["boost"].without_graph = True
        self.options["boost"].without_graph_parallel = True
        self.options["boost"].without_json = True
        self.options["boost"].without_log = True
        self.options["boost"].without_math = True
        self.options["boost"].without_mpi = True
        self.options["boost"].without_nowide = True
        self.options["boost"].without_python = True
        self.options["boost"].without_serialization = True
        self.options["boost"].without_stacktrace = True
        self.options["boost"].without_test = True
        self.options["boost"].without_timer = True
        self.options["boost"].without_type_erasure = True
        self.options["boost"].without_wave = True

    def imports(self):
        self.copy("*.dylib", "Frameworks", "lib")
