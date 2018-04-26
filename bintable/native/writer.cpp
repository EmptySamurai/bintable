#include "common.h"
#include "writer.h"
#include "operations/tableoperations.h"
#include "Python.h"
#include "exceptions.h"



using namespace NAMESPACE_BINTABLE;

BaseOperation* FromPythonOperationsSelector::select_write_operation(ReadWriteSpecification& spec) {
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawWriteOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    } else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8) {
        auto fixed_op = new FromFixedLengthStringWriteOperation();
        fixed_op->size = DATATYPE_ELEMENT_SIZE[spec.type];
        fixed_op->maxlen = spec.maxlen;
        op = fixed_op;
    } else if (spec.type == BINTABLE_OBJECT) {
        op = new FromPyObjectWriteOperation();
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation* FromPythonOperationsSelector::select_skip_operation(ReadWriteSpecification& spec) {
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    } else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8) {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = spec.maxlen;
        op = raw_op;
    } else if (spec.type == BINTABLE_OBJECT) {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = sizeof(PyObject*);
        op = raw_op;
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation* ToPythonOperationsSelector::select_write_operation(ReadWriteSpecification& spec) {
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawWriteOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    } else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8) {
        auto fixed_op = new ToFixedLengthStringWriteOperation();
        fixed_op->size = DATATYPE_ELEMENT_SIZE[spec.type];
        fixed_op->maxlen = spec.maxlen;
        op = fixed_op;
    } else if (spec.type == BINTABLE_OBJECT) {
        op = new ToPyObjectWriteOperation();
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation* ToPythonOperationsSelector::select_skip_operation(ReadWriteSpecification& spec) {
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    } else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8) {
        op = new BinTableStringSkipOperation();
    } else if (spec.type == BINTABLE_OBJECT) {
        op = new BinTableStringSkipOperation();
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation* Optimizer::optimize(BaseOperation * operation) {
    if (operation->operation_type == "SEQUENCE") {
        operation = optimize_sequence(reinterpret_cast<SequenceOperation*>(operation));
    } else if (operation->operation_type == "LOOP") {
        operation = optimize_loop(reinterpret_cast<LoopOperation*>(operation));
    }

    return operation;
}

BaseOperation* Optimizer::optimize_sequence(SequenceOperation* sequence) {
    auto &ops = sequence->operations;

    auto n_ops = ops.size();

    // Optimize each operation
    for (auto i=0; i<ops.size(); i++) {
        ops[i] = optimize(ops[i]);
    }

    // Merge RAW operations
    for (auto i=0; i<ops.size(); i++) {

        if ((ops[i]->operation_type == "RAW_SKIP") || (ops[i]->operation_type == "RAW_WRITE")) {
            auto op = reinterpret_cast<RawOperation*>(ops[i]);
            auto j = i+1;

            for (;j<ops.size(); j++) {
                if (ops[j]->operation_type == op->operation_type) {
                    auto op_to_merge = reinterpret_cast<RawOperation*>(ops[j]);
                    if ((op->input_stream == op_to_merge->input_stream) && 
                    (op->output_stream == op_to_merge->output_stream)) {
                        op->n_bytes+=op_to_merge->n_bytes;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            for (auto t = i+1; t<j; t++) {
                delete ops[t];
            }
            ops.erase(ops.begin()+i+1, ops.begin()+j);
        }
    }

    //If one element left return it
    if (ops.size() == 1) {
        auto op = ops[0];
        ops.clear();
        delete sequence;
        return op;
    }

    return sequence;
}


BaseOperation* Optimizer::optimize_loop(LoopOperation* loop) {
    loop->operation = optimize(loop->operation);
    if ((loop->operation->operation_type == "RAW_SKIP") || (loop->operation->operation_type == "RAW_WRITE")) {
        auto op_in_loop = reinterpret_cast<RawOperation*>(loop->operation);
        op_in_loop->n_bytes *= loop->n_iter;
        loop->operation = nullptr;
        delete loop;
        return op_in_loop;
    } 
    
    return loop;
}


Writer::Writer(OperationsSelector* selector, bool create_sequence) : selector(selector), responsible_for_sequence(create_sequence) {
    if (create_sequence) {
        sequence = new SequenceOperation();
    }
}

Writer::Writer(OperationsSelector* selector) : Writer::Writer(selector, true) {

}

void Writer::add_operation(BaseOperation* operation) {
    reinterpret_cast<SequenceOperation*>(sequence)->operations.push_back(operation);
}

void Writer::write(ReadWriteSpecification& spec) {
    auto op = selector->select_write_operation(spec);
    add_operation(op); 
}

void Writer::skip(ReadWriteSpecification& spec) {
    auto op = selector->select_skip_operation(spec);
    add_operation(op); 
}

Writer Writer::loop(uint64_t n_iter) {
    Writer loop_writer(selector, false);
    auto loop_sequence_op = new SequenceOperation();
    loop_writer.sequence = loop_sequence_op;

    auto loop_op = new LoopOperation();
    loop_op->operation=loop_sequence_op;
    loop_op->n_iter = n_iter;
    add_operation(loop_op); 

    return loop_writer;
}

void print_op_tree(BaseOperation* operation, std::string prefix = "") {
    if (operation->operation_type == "SEQUENCE") {
        auto seq_operation = reinterpret_cast<SequenceOperation*>(operation);
        PRINT(prefix<<seq_operation->operation_type);
        for (auto op : seq_operation->operations) {
            print_op_tree(op, "\t"+prefix);
        }
    } else if (operation->operation_type == "LOOP") {
        auto loop_operation = reinterpret_cast<LoopOperation*>(operation);
        PRINT(prefix<<loop_operation->operation_type<<" "<<loop_operation->n_iter<<"x");
        print_op_tree(loop_operation->operation,"\t"+prefix);
    } else {
        PRINT(prefix<<operation->operation_type);
    }
}

void Writer::run() {
    if (responsible_for_sequence) {
        Optimizer optimizer;
        sequence = optimizer.optimize(sequence);
        print_op_tree(sequence);
        (*sequence)();
    } else {
        throw BinTableException("Running non root writers is not permitted");
    }
}

Writer::~Writer() {
    if (responsible_for_sequence) {
        delete sequence;
    }
}