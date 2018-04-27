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

FromFixedLengthStringWriteOperation::FromFixedLengthStringWriteOperation(uint8_t size, uint32_t maxlen) {
    this->size = size;
    this->maxlen = maxlen;
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
    delete[] buffer;
}


ToFixedLengthStringWriteOperation::ToFixedLengthStringWriteOperation(uint8_t size, uint32_t maxlen) : zero_stream(ConstantInputStream(0)) {
    this->size = size;
    this->maxlen = maxlen;
    operation_type = "FROM_FIXED_LENGTH_STRING_WRITE";
}

void ToFixedLengthStringWriteOperation::operator()() {
    uint32_t len;
    BinTableString::read_to_stream(*input_stream, *output_stream, len);
    output_stream->write(zero_stream, maxlen-len);
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