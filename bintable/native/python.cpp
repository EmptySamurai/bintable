#include "common.h"
#include "bintable.h"
#include "pythonutils.h"
#include "numpy.h"
#include "exceptions.h"
#include <vector>
#include <string>
#include <cstdio>
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
        if (!append)
        {
            remove(path.c_str());
        }
        throw;
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

py::dict read_header_interface(const std::string &path)
{
    BinTableHeader header = read_header(path);

    py::dict dict;
    dict["version"] = header.version;
    dict["n_rows"] = header.n_rows;
    dict["n_columns"] = header.n_columns;

    py::dict cols_dict;
    for (auto it = header.columns.begin(); it != header.columns.end(); it++)
    {
        auto &col = *it;
        cols_dict[col.name->to_string().c_str()] = DATATYPE_NAME[col.type];
    }
    dict["columns"] = cols_dict;

    return dict;
}

PYBIND11_MODULE(native, m)
{
    m.doc() = "Bintable native code"; // optional module docstring
    
    m.attr("USE_LITTLE_ENDIAN") = py::int_(USE_LITTLE_ENDIAN);
    m.def("write_table", &write_table_interface, "Function to write table");
    m.def("read_table", &read_table_interface, "Function to read table");
    m.def("read_header", &read_header_interface, "Function to read header");

    py::register_exception<BinTableException>(m, "BinTableException");
}