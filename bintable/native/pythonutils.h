#pragma once
#include "common.h"
#include "Python.h"
#include "tablestring.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

BinTableString* python_string_to_table_string(PyObject* obj) {
    Py_ssize_t size = 0;
    char* data = PyUnicode_AsUTF8AndSize(obj, &size);

    auto result = new BinTableString();
    result->data = data;
    result->size = size;

    return result;
}

PyObject* table_string_to_python_string(BinTableString* str) {
    const char *errors = NULL;
    return PyUnicode_DecodeUTF8(str->data, str->size, errors);
}


NAMESPACE_END(NAMESPACE_BINTABLE)
