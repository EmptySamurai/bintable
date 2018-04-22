#pragma once
#include "common.h"
#include "Python.h"
#include "tablestring.h"


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

void python_string_to_table_string(PyObject* obj, BinTableString& out);

PyObject* table_string_to_python_string(BinTableString& str);

void python_object_to_table_string(PyObject* obj, BinTableString& out);

PyObject* table_string_to_python_object(BinTableString& str);

NAMESPACE_END(NAMESPACE_BINTABLE)
