#pragma once
#include <numpy/arrayobject.h>
#include "bintable.h"
#include "common.h"


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

void column_data_from_numpy_array(PyArrayObject *arr, BinTableColumnData& out);
PyArrayObject* numpy_array_to_column_data(BinTableColumnData& columnData);

NAMESPACE_END(NAMESPACE_BINTABLE)

