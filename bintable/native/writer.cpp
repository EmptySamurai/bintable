#include "common.h"
#include "writer.h"
#include "operations/tableoperations.h"
#include "Python.h"
#include "exceptions.h"

using namespace NAMESPACE_BINTABLE;

BaseOperation *FromPythonOperationsSelector::select_write_operation(ReadWriteSpecification &spec)
{
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawWriteOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    }
    else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8)
    {
        auto fixed_op = new FromFixedLengthStringWriteOperation();
        fixed_op->size = DATATYPE_ELEMENT_SIZE[spec.type];
        fixed_op->maxlen = spec.maxlen;
        op = fixed_op;
    }
    else if (spec.type == BINTABLE_OBJECT)
    {
        op = new FromPyObjectWriteOperation();
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation *FromPythonOperationsSelector::select_skip_operation(ReadWriteSpecification &spec)
{
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    }
    else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8)
    {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = spec.maxlen;
        op = raw_op;
    }
    else if (spec.type == BINTABLE_OBJECT)
    {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = sizeof(PyObject *);
        op = raw_op;
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation *ToPythonOperationsSelector::select_write_operation(ReadWriteSpecification &spec)
{
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawWriteOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    }
    else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8)
    {
        auto fixed_op = new ToFixedLengthStringWriteOperation();
        fixed_op->size = DATATYPE_ELEMENT_SIZE[spec.type];
        fixed_op->maxlen = spec.maxlen;
        op = fixed_op;
    }
    else if (spec.type == BINTABLE_OBJECT)
    {
        op = new ToPyObjectWriteOperation();
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation *ToPythonOperationsSelector::select_skip_operation(ReadWriteSpecification &spec)
{
    ReadWriteOperation *op;
    if (is_basic_bintable_datatype(spec.type))
    {
        auto raw_op = new RawSkipOperation();
        raw_op->n_bytes = DATATYPE_ELEMENT_SIZE[spec.type];
        op = raw_op;
    }
    else if (spec.type == BINTABLE_UTF32 || spec.type == BINTABLE_UTF8)
    {
        op = new BinTableStringSkipOperation();
    }
    else if (spec.type == BINTABLE_OBJECT)
    {
        op = new BinTableStringSkipOperation();
    }
    op->input_stream = spec.input_stream;
    op->output_stream = spec.output_stream;

    return op;
}

BaseOperation *Optimizer::optimize(BaseOperation *operation)
{
    if (operation->operation_type == "SEQUENCE")
    {
        operation = optimize_sequence(reinterpret_cast<SequenceOperation *>(operation));
    }
    else if (operation->operation_type == "LOOP")
    {
        operation = optimize_loop(reinterpret_cast<LoopOperation *>(operation));
    }

    return operation;
}

BaseOperation *Optimizer::optimize_sequence(SequenceOperation *sequence)
{
    auto &ops = sequence->operations;

    auto n_ops = ops.size();

    // Optimize each operation
    for (auto i = 0; i < ops.size(); i++)
    {
        ops[i] = optimize(ops[i]);
    }

    // Delete noops
    delete_noop(ops);

    // Moving operations of subsequences into current sequence
    insert_subsequences(ops);

    // Merge RAW operations
    merge_raw(ops);

    //If one element left replace sequence by it
    if (ops.size() == 1)
    {
        auto op = ops[0];
        ops.clear();
        delete sequence;
        return op;
    }
    //If no element left replace sequence noop
    else if (ops.size() == 0)
    {
        delete sequence;
        return new NoOperation();
    }

    return sequence;
}

void Optimizer::delete_noop(std::vector<BaseOperation *> &ops)
{
    for (auto i = 0; i < ops.size(); i++)
    {
        if (ops[i]->operation_type == "NOOP")
        {
            auto j = i + 1;
            for (; j < ops.size(); j++)
            {
                if (ops[j]->operation_type != "NOOP")
                {
                    break;
                }
            }

            for (auto t = i; t < j; t++)
            {
                delete ops[t];
            }
            ops.erase(ops.begin() + i, ops.begin() + j);
            i--;
        }
    }
}

void Optimizer::insert_subsequences(std::vector<BaseOperation *> &ops)
{
    for (auto i = 0; i < ops.size(); i++)
    {
        if (ops[i]->operation_type == "SEQUENCE")
        {
            auto seq_op = reinterpret_cast<SequenceOperation *>(ops[i]);
            auto seq_size = seq_op->operations.size();
            ops.erase(ops.begin()+i, ops.begin()+i+1);
            ops.insert(ops.begin()+i, seq_op->operations.begin(), seq_op->operations.end());
            i+=seq_size-1;

            seq_op->operations.clear();
            delete seq_op;
        }
    }
}

void Optimizer::merge_raw(std::vector<BaseOperation *> &ops)
{
    for (auto i = 0; i < ops.size(); i++)
    {

        if ((ops[i]->operation_type == "RAW_SKIP") || (ops[i]->operation_type == "RAW_WRITE"))
        {
            auto op = reinterpret_cast<RawOperation *>(ops[i]);
            auto j = i + 1;

            for (; j < ops.size(); j++)
            {
                if (ops[j]->operation_type == op->operation_type)
                {
                    auto op_to_merge = reinterpret_cast<RawOperation *>(ops[j]);
                    if ((op->input_stream == op_to_merge->input_stream) &&
                        (op->output_stream == op_to_merge->output_stream))
                    {
                        op->n_bytes += op_to_merge->n_bytes;
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }

            for (auto t = i + 1; t < j; t++)
            {
                delete ops[t];
            }
            ops.erase(ops.begin() + i + 1, ops.begin() + j);
        }
    }
}

BaseOperation *Optimizer::optimize_loop(LoopOperation *loop)
{
    loop->operation = optimize(loop->operation);
    bool extract_op = false;
    // If RAW - replace loop by one large RAW
    if ((loop->operation->operation_type == "RAW_SKIP") || (loop->operation->operation_type == "RAW_WRITE"))
    {
        reinterpret_cast<RawOperation *>(loop->operation)->n_bytes *= loop->n_iter;
        extract_op = true;
    }
    // If noop - replace loop by noop
    else if (loop->operation->operation_type == "NOOP")
    {
        extract_op = true;
    }
    // If 1 iteration - replace by internal op 
    else if (loop->n_iter == 1) {
        extract_op = true;
    } 
    // If no iterations, then replace by noop
    else if (loop->n_iter ==0) {
        delete loop;
        return new NoOperation();
    }

    if (extract_op)
    {
        auto op_in_loop = loop->operation;
        loop->operation = nullptr;
        delete loop;
        return op_in_loop;
    }

    return loop;
}

Writer::Writer(OperationsSelector *selector, bool create_sequence) : selector(selector), responsible_for_sequence(create_sequence)
{
    if (create_sequence)
    {
        sequence = new SequenceOperation();
    }
}

Writer::Writer(OperationsSelector *selector) : Writer::Writer(selector, true)
{
}

void Writer::add_operation(BaseOperation *operation)
{
    sequence->operations.push_back(operation);
}

void Writer::write(ReadWriteSpecification &spec)
{
    auto op = selector->select_write_operation(spec);
    add_operation(op);
}

void Writer::skip(ReadWriteSpecification &spec)
{
    auto op = selector->select_skip_operation(spec);
    add_operation(op);
}

Writer Writer::loop(uint64_t n_iter)
{
    Writer loop_writer(selector, false);
    auto loop_sequence_op = new SequenceOperation();
    loop_writer.sequence = loop_sequence_op;

    auto loop_op = new LoopOperation();
    loop_op->operation = loop_sequence_op;
    loop_op->n_iter = n_iter;
    add_operation(loop_op);

    return loop_writer;
}

void print_op_tree(BaseOperation *operation, std::string prefix = "")
{
    if (operation->operation_type == "SEQUENCE")
    {
        auto seq_operation = reinterpret_cast<SequenceOperation *>(operation);
        PRINT(prefix << seq_operation->operation_type);
        for (auto op : seq_operation->operations)
        {
            print_op_tree(op, "\t" + prefix);
        }
    }
    else if (operation->operation_type == "LOOP")
    {
        auto loop_operation = reinterpret_cast<LoopOperation *>(operation);
        PRINT(prefix << loop_operation->operation_type << " " << loop_operation->n_iter << "x");
        print_op_tree(loop_operation->operation, "\t" + prefix);
    }
    else
    {
        PRINT(prefix << operation->operation_type);
    }
}

void Writer::run(bool optimize)
{
    if (responsible_for_sequence)
    {
        BaseOperation* op_to_run;
        if (optimize) {
            Optimizer optimizer;
            auto optimized_op = optimizer.optimize(sequence);

            sequence = new SequenceOperation();
            add_operation(optimized_op);
            
            op_to_run = optimized_op;
        } else {
            op_to_run = sequence; 
        }

        print_op_tree(op_to_run);
        (*op_to_run)();
    }
    else
    {
        throw BinTableException("Running non root writers is not permitted");
    }
}

Writer::~Writer()
{
    if (responsible_for_sequence)
    {
        delete sequence;
    }
}