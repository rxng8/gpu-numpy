#include <pybind11/pybind11.h>



int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(gnpc, m) {
    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
}
