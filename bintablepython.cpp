#include "Python.h"
#include <numpy/arrayobject.h>


static PyObject* write_table_python(PyObject *self, PyObject *args) {
    int nTags, nDocs, nTerms;
    PyObject * arraysList;
    PyObject * namesList;


    if (!PyArg_ParseTuple(args, "OO", &arraysList, &namesList))
        return NULL;

    long nValues = PyList_Size(arraysList); //TODO: Check if lists, or replace by len
    for (long i = 0; i < nValues; i++) {
        PyArrayObject * array = PyFloat_AsDouble(PyList_GetItem(arraysList, i));
    }
}


static PyMethodDef TableMethods[] = {

        {"write_table", write_table_python, METH_VARARGS,
                "Execute a shell command."},

        {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef bintablenative = {
        PyModuleDef_HEAD_INIT,
        "native", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1, /* size of per-interpreter state of the module,
			  or -1 if the module keeps state in global variables. */
        TableMethods
};

PyMODINIT_FUNC
PyInit_native(void) {
    return PyModule_Create(&bintablenative);
}
