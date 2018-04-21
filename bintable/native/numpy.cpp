#include "numpy.h"
#include <map>
#include <iostream>

using namespace NAMESPACE_BINTABLE;

void * enable_numpy_support()
{
    import_array();
    return nullptr;
}

const static void* numpy_initialized = enable_numpy_support();

std::map<tabledatatype, int> table_to_numpy_types = {{BINTABLE_BOOL, NPY_BOOL},
                                                     {BINTABLE_INT8, NPY_INT8},
                                                     {BINTABLE_UINT8, NPY_UINT8},
                                                     {BINTABLE_INT16, NPY_INT16},
                                                     {BINTABLE_UINT16, NPY_UINT16},
                                                     {BINTABLE_INT32, NPY_INT32},
                                                     {BINTABLE_UINT32, NPY_UINT32},
                                                     {BINTABLE_INT64, NPY_INT64},
                                                     {BINTABLE_UINT32, NPY_UINT64},
                                                     {BINTABLE_FLOAT32, NPY_FLOAT32},
                                                     {BINTABLE_FLOAT64, NPY_FLOAT64},
                                                     {BINTABLE_UTF8, NPY_STRING},
                                                     {BINTABLE_UTF32, NPY_UNICODE}};

std::map<tabledatatype, int> numpy_to_table_types = {{NPY_BOOL, BINTABLE_BOOL},
                                                {NPY_INT8, BINTABLE_INT8},
                                                {NPY_UINT8, BINTABLE_UINT8},
                                                {NPY_INT16, BINTABLE_INT16},
                                                {NPY_UINT16, BINTABLE_UINT16},
                                                {NPY_INT32, BINTABLE_INT32},
                                                {NPY_UINT32, BINTABLE_UINT32},
                                                {NPY_INT64, BINTABLE_INT64},
                                                {NPY_UINT64, BINTABLE_UINT32},
                                                {NPY_FLOAT32, BINTABLE_FLOAT32},
                                                {NPY_FLOAT64, BINTABLE_FLOAT64},
                                                {NPY_STRING, BINTABLE_UTF8},
                                                {NPY_UNICODE, BINTABLE_UTF32}};

void NAMESPACE_BINTABLE::column_data_from_numpy_array(PyArrayObject *arr, BinTableColumnData &out)
{
    out.size = PyArray_SIZE(arr);
    out.data = PyArray_BYTES(arr);
    out.type = numpy_to_table_types[PyArray_TYPE(arr)];

    if (out.type == BINTABLE_UTF32 || out.type == BINTABLE_UTF8) {
        out.maxlen = arr->descr->elsize;
    } else {
        out.maxlen = DATATYPE_ELEMENT_SIZE[out.type];
    }; 
}

PyObject* NAMESPACE_BINTABLE::numpy_array_from_column_data(BinTableColumnData& columnData) {
    npy_intp* dims = new npy_intp[1];
    dims[0] = columnData.size;
    
    PyArray_Descr* descr = PyArray_DescrNewFromType(table_to_numpy_types[columnData.type]);
    descr->byteorder = '<'; 

    if (columnData.type==BINTABLE_UTF32 || columnData.type==BINTABLE_UTF8) {
        descr->elsize = columnData.maxlen; 
    };

    PyObject* result =  PyArray_NewFromDescr( &PyArray_Type, descr, 1, dims, NULL, columnData.data, NPY_ARRAY_CARRAY, NULL);

    delete dims;

    return result;
}