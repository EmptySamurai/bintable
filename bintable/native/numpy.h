#pragma once
#include <numpy/arrayobject.h>
#include "bintable.h"
#include "common.h"


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

void column_data_from_numpy_array(PyArrayObject *arr, BinTableColumnData& out);
PyObject* numpy_array_from_column_data(BinTableColumnData& columnData);

NAMESPACE_END(NAMESPACE_BINTABLE)

