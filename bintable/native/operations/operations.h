#pragma once
#include "common.h"
#include "streams/streams.h"
#include <string>
#include <vector>

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class BaseOperation
{
  public:
    std::string operation_type;
    virtual void operator()() = 0;
    virtual ~BaseOperation() = default;
};

class SequenceOperation : public BaseOperation
{
  public:
    SequenceOperation();
    void operator()() override;
    ~SequenceOperation() override;
    std::vector<BaseOperation *> operations;
};

class LoopOperation : public BaseOperation
{
  public:
    LoopOperation();
    void operator()() override;
    ~LoopOperation() override;
    BaseOperation *operation;
    uint64_t n_iter;
};

class ReadWriteOperation : public BaseOperation
{
  public:
    InputStream* input_stream = nullptr;
    OutputStream* output_stream = nullptr;
};

class RawOperation : public ReadWriteOperation {
    public:
        uint64_t n_bytes;
};

class RawSkipOperation : public RawOperation {
    public:
        RawSkipOperation();
        void operator()() override;
};

class RawWriteOperation : public RawOperation {
    public:
        RawWriteOperation();
        void operator()() override;
};

NAMESPACE_END(NAMESPACE_BINTABLE)
