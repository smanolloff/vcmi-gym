from conan import ConanFile
from conan.tools.cmake import CMakeToolchain

required_conan_version = ">=2.13.0"

class Connector(ConanFile):
    generators = "CMakeDeps"
    settings = "os", "compiler", "build_type", "arch"
    requires = ["pybind11/[~2.12.0]", "boost/[^1.74 <1.87]"]
    default_options = {"boost/*:shared": True}

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
