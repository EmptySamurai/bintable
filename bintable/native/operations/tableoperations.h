#pragma once
#include "common.h"
#include "operations/operations.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class BinTableStringSkipOperation :public ReadWriteOperation {
    public:
        BinTableStringSkipOperation();
        void operator()() override;
};

class FixedLengthStringOperation : public ReadWriteOperation {
    public:
        uint8_t size;
        uint32_t maxlen;
};

class FromFixedLengthStringWriteOperation : public FixedLengthStringOperation {
    public:
        FromFixedLengthStringWriteOperation();
        void operator()() override;
        ~FromFixedLengthStringWriteOperation() override;
    private:
        char* buffer;
};

//From fixed string skip is RAW

class ToFixedLengthStringWriteOperation : public FixedLengthStringOperation {
    public:
        ToFixedLengthStringWriteOperation();
        void operator()() override;
        ~ToFixedLengthStringWriteOperation() override;
    private:
        char* fill_buffer;
};

//To fixed string skip is BinTableString skip


class FromPyObjectWriteOperation : public ReadWriteOperation {
    public:
        FromPyObjectWriteOperation();
        void operator()() override;
};

//From object skip is RAW


class ToPyObjectWriteOperation : public ReadWriteOperation {
    public:
        ToPyObjectWriteOperation();
        void operator()() override;
};

//To object skip is BinTableString skip

NAMESPACE_END(NAMESPACE_BINTABLE)



