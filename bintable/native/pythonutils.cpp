#include "common.h"
#include "Python.h"
#include  <limits>
#include "exceptions.h"
#include "tablestring.h"
#include "pythonutils.h"


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

void python_string_to_table_string(PyObject* obj, BinTableString& out) {
    Py_ssize_t size = 0;
    char* data = PyUnicode_AsUTF8AndSize(obj, &size);

    out.set(data, size, false);
}

PyObject* table_string_to_python_string(BinTableString& str) {
    const char *errors = NULL;
    return PyUnicode_DecodeUTF8(str.get_data(), str.get_size(), errors);
}

char _NULL_DATA[] = {0};

void _to_null_string(BinTableString& out) {
    out.set(_NULL_DATA, 1, false);
}

void python_object_to_table_string(PyObject* obj, BinTableString& out) {

    if (PyUnicode_Check(obj)) {
        python_string_to_table_string(obj, out);
    } else if (PyFloat_Check(obj)) {
        double value = PyFloat_AsDouble(obj);
        if (std::isnan(value)) {
            _to_null_string(out);
        } else {
            throw WrongPythonObjectException("Object array contains non NaN floats");
        }
    } else if (obj == Py_None) {
        _to_null_string(out);
    } else {
        throw WrongPythonObjectException("Object array contains non string, NaN or None objects");
    }
}

PyObject* table_string_to_python_object(BinTableString& str) {
    const char *errors = NULL;
    if ((str.get_size() == 1) && (str.get_data()[0]==0)) {
        return PyFloat_FromDouble(std::numeric_limits<double>::quiet_NaN());
    } else {
        return table_string_to_python_string(str);
    }
}


NAMESPACE_END(NAMESPACE_BINTABLE)
