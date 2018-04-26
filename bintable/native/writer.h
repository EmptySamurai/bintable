#pragma once
#include "common.h"
#include "streams/streams.h"
#include "operations/operations.h"
#include "types.h"
#include <vector>

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

struct ReadWriteSpecification {
    InputStream* input_stream;
    OutputStream* output_stream;
    tabledatatype type;
    uint32_t maxlen;
};

class OperationsSelector {
    public:
        virtual BaseOperation* select_write_operation(ReadWriteSpecification& spec)=0;
        virtual BaseOperation* select_skip_operation(ReadWriteSpecification& spec)=0;
};

class FromPythonOperationsSelector : public OperationsSelector {
    public:
        BaseOperation* select_write_operation(ReadWriteSpecification& spec) override;
        BaseOperation* select_skip_operation(ReadWriteSpecification& spec) override;
};

class ToPythonOperationsSelector : public OperationsSelector {
    public:
        BaseOperation* select_write_operation(ReadWriteSpecification& spec) override;
        BaseOperation* select_skip_operation(ReadWriteSpecification& spec) override;
};

class Optimizer {
    public:
        BaseOperation* optimize(BaseOperation * operation);
    private:
        BaseOperation* optimize_sequence(SequenceOperation* sequence);
        void delete_noop(std::vector<BaseOperation*>& operations);
        void merge_raw(std::vector<BaseOperation*>& operations);
        BaseOperation* optimize_loop(LoopOperation* loop);
};

class Writer {
    public:
        Writer(OperationsSelector* selector);
        void write(ReadWriteSpecification& spec);
        void skip(ReadWriteSpecification& spec);
        Writer loop(uint64_t n_iter);
        void run();
        ~Writer();

    private:
        Writer(OperationsSelector* selector, bool create_sequence);
        void add_operation(BaseOperation* operation);
        OperationsSelector* selector;
        bool responsible_for_sequence;
        BaseOperation* sequence;
};

NAMESPACE_END(NAMESPACE_BINTABLE)
