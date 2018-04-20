#include "common.h"
#include "bintable.h"
#include "pythonutils.h"
#include "numpy.h"
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <numpy/arrayobject.h>


namespace py = pybind11;

using namespace NAMESPACE_BINTABLE;

void write_table_interface(const py::dict columnsDict, const std::string &path, bool append) {
    std::vector<BinTableColumnData* > columns;
    columns.reserve(columnsDict.size());
    for (auto name_data : columnsDict) {
        BinTableColumnData * column = new BinTableColumnData();
        column->name = python_string_to_table_string(name_data.first.ptr());
        column_data_from_numpy_array((PyArrayObject*)name_data.second.ptr(), *column);
        columns.push_back(column);
    }
    write_table(columns, path, append);

    for (BinTableColumnData* column : columns) {
        delete column->name;
        delete column;
    }
}




PYBIND11_MODULE(native, m)
{
    m.doc() = "Bintable native code"; // optional module docstring

    m.def("write_table", &write_table_interface, "Function to write table");
}