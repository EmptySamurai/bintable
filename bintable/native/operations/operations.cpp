#include "common.h"
#include "operations/operations.h"

using namespace NAMESPACE_BINTABLE;

// NOOP

NoOperation::NoOperation()
{
    operation_type = "NOOP";
}

void NoOperation::operator()()
{
    
}

// COLLECTION OPERATION

SequenceOperation::SequenceOperation()
{
    operation_type = "SEQUENCE";
}

void SequenceOperation::operator()()
{
    for (auto op : operations)
    {
        (*op)();
    }
}

SequenceOperation::~SequenceOperation()
{
    for (auto op : operations)
    {
        delete op;
    }
}

// LOOP OPERATION

LoopOperation::LoopOperation() : operation(nullptr)
{
    operation_type = "LOOP";
}

void LoopOperation::operator()()
{
    for (uint64_t i = 0; i < n_iter; i++)
    {
        (*operation)();
    }
}

LoopOperation::~LoopOperation()
{
    delete operation;
}

//RAW SKIP OPERATION

RawSkipOperation::RawSkipOperation(uint64_t n_bytes)
{
    this->n_bytes = n_bytes;
    operation_type = "RAW_SKIP";
}

void RawSkipOperation::operator()()
{
    input_stream->skip(n_bytes);
}

//RAW WRITE OPERAION

RawWriteOperation::RawWriteOperation(uint64_t n_bytes)
{
    this->n_bytes = n_bytes;
    operation_type = "RAW_WRITE";
}

void RawWriteOperation::operator()()
{
    output_stream->write(*input_stream, n_bytes);
}
