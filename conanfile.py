from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout


class NexusDataConan(ConanFile):
    name = "nexusdata"
    version = "1.0.0"
    package_type = "library"
    license = "Apache-2.0"
    author = "Nexus"
    url = "https://github.com/nexus-ai/nexusdata"
    description = "High-performance C++20 AI data loading library"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "with_cuda": [True, False],
        "with_sqlite": [True, False],
    }
    default_options = {
        "shared": False,
        "with_cuda": False,
        "with_sqlite": False,
    }
    exports_sources = (
        "CMakeLists.txt",
        "cmake/*",
        "include/*",
        "src/*",
        "cuda/*",
        "third_party/*",
    )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["NEXUSDATA_BUILD_TESTS"] = False
        tc.variables["NEXUSDATA_BUILD_EXAMPLES"] = False
        tc.variables["NEXUSDATA_BUILD_BENCH"] = False
        tc.variables["NEXUSDATA_BUILD_FUZZ"] = False
        tc.variables["NEXUSDATA_WITH_CUDA"] = self.options.with_cuda
        tc.variables["NEXUSDATA_WITH_SQLITE"] = self.options.with_sqlite
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["nexusdata"]
        self.cpp_info.set_property("cmake_file_name", "NexusData")
        self.cpp_info.set_property("cmake_target_name", "nexusdata::nexusdata")
