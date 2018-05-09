#pragma once
#include "common.h"
#include "operations/operations.h"
#include "streams/constantstream.h"
#include "tablestring.h"

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
        FromFixedLengthStringWriteOperation(uint8_t size, uint32_t maxlen);
        void operator()() override;
        ~FromFixedLengthStringWriteOperation() override;
    private:
        char* buffer;
};

//From fixed string skip is RAW

class ToFixedLengthStringWriteOperation : public FixedLengthStringOperation {
    public:
        ToFixedLengthStringWriteOperation(uint8_t size, uint32_t maxlen);
        void operator()() override;
    private:
        ConstantInputStream zero_stream;
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
    
    private:
        BinTableString temp_string;
};

//To object skip is BinTableString skip

NAMESPACE_END(NAMESPACE_BINTABLE)



