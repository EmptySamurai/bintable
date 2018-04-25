#include "common.h"
#include "operations/tableoperations.h"
#include "tablestring.h"
#include "Python.h"
#include "pythonutils.h"


using namespace NAMESPACE_BINTABLE;

BinTableStringSkipOperation::BinTableStringSkipOperation() {
    operation_type = "BINTABLE_SKIP";
}

void BinTableStringSkipOperation::operator()() {
    uint32_t size;
    BinTableString::skip(*input_stream, size);
}

FromFixedLengthStringWriteOperation::FromFixedLengthStringWriteOperation() {
    operation_type = "FROM_FIXED_LENGTH_STRING_WRITE";
    buffer = new char[maxlen];
}

void FromFixedLengthStringWriteOperation::operator()() {
    input_stream->read(buffer, maxlen);
    char *data_start = buffer;
    char *data_end = data_start + maxlen;
    uint32_t len = maxlen;
    for (uint32_t i = 0; i < maxlen; i += size)
    {
        char notNull = 0;
        for (uint8_t j = 1; j <= size; j++)
        {
            notNull = notNull || data_end[-j];
        }
        if (notNull)
        {
            break;
        }
        else
        {
            len -= size;
            data_end -= size;
        }
    }
    BinTableString temp_string;
    temp_string.data = data_start;
    temp_string.size = len;
    temp_string.write(*output_stream);
}

FromFixedLengthStringWriteOperation::~FromFixedLengthStringWriteOperation() {
    delete buffer;
}


ToFixedLengthStringWriteOperation::ToFixedLengthStringWriteOperation() {
    operation_type = "FROM_FIXED_LENGTH_STRING_WRITE";
    fill_buffer = new char[maxlen];
    std::fill(fill_buffer, fill_buffer+maxlen, 0);
}

void ToFixedLengthStringWriteOperation::operator()() {
    uint32_t size;
    BinTableString::read_to_stream(*input_stream, *output_stream, size);
    output_stream->write(fill_buffer, maxlen-size);
}

ToFixedLengthStringWriteOperation::~ToFixedLengthStringWriteOperation() {
    delete fill_buffer;
}


FromPyObjectWriteOperation::FromPyObjectWriteOperation() {
    operation_type = "FROM_PYOBJECT_WRITE";
}

void FromPyObjectWriteOperation::operator()() {
    PyObject* obj;
    input_stream->read_primitive(obj);
    BinTableString temp_string;
    python_object_to_table_string(obj, temp_string);
    temp_string.write(*output_stream);
}


ToPyObjectWriteOperation::ToPyObjectWriteOperation() {
    operation_type = "TO_PYOBJECT_WRITE";
}

void ToPyObjectWriteOperation::operator()() {
    BinTableString table_string(*input_stream);
    output_stream->write_primitive(table_string_to_python_object(table_string));
}