from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.apple import is_apple_os
from conan.tools.build import cross_building
from conan.tools.cmake import CMakeDeps, CMakeToolchain
from conans import tools

required_conan_version = ">=1.51.3"


class Connector(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = ["pybind11/[~2.7.1]"]

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
            # "Release",
            # "RelWithDebInfo",
        ]

        for config in configs:
            print(f"generating CMakeDeps for {config}")
            deps.configuration = config
            deps.generate()

    def imports(self):
        self.copy("*.dylib", "Frameworks", "lib")
