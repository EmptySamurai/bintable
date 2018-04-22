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

void _delete_columns(std::vector<BinTableColumnData *> &columns)
{
    for (BinTableColumnData *column : columns)
    {
        delete column->name;
        delete column;
    }
}

void write_table_interface(const py::dict columns_dict, const std::string &path, bool append)
{
    std::vector<BinTableColumnData *> columns;
    columns.reserve(columns_dict.size());
    try
    {
        for (auto name_data : columns_dict)
        {
            BinTableColumnData *column = new BinTableColumnData();
            column->name = new BinTableString();
            python_string_to_table_string(name_data.first.ptr(), *(column->name));
            column_data_from_numpy_array((PyArrayObject *)name_data.second.ptr(), *column);
            columns.push_back(column);
        }
        write_table(columns, path, append);
    }
    catch (std::exception ex)
    {
        _delete_columns(columns);
        throw ex;
    }
    _delete_columns(columns);
}

py::dict read_table_interface(const std::string &path)
{
    std::vector<BinTableColumnData *> columns;
    read_table(path, columns);
    py::dict dict;

    for (auto col : columns)
    {
        auto key = py::reinterpret_steal<py::str>(table_string_to_python_string(*(col->name)));
        auto value = py::reinterpret_steal<py::object>(numpy_array_from_column_data(*col));
        dict[key] = value;
    }

    _delete_columns(columns);
    return dict;
}

PYBIND11_MODULE(native, m)
{
    m.doc() = "Bintable native code"; // optional module docstring

    m.def("write_table", &write_table_interface, "Function to write table");
    m.def("read_table", &read_table_interface, "Function to read table");
}