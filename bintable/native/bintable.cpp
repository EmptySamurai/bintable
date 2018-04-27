#include "bintable.h"
#include "pythonutils.h"

#include "streams/streams.h"
#include "streams/bufferedstreams.h"
#include "streams/memorystreams.h"


#include "writer.h"

#include "Python.h"
#include <ostream>
#include <fstream>
#include <functional>
#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <iostream>

using namespace NAMESPACE_BINTABLE;

const int BUFFER_SIZE = 100000;

BinTableHeader _create_header(std::vector<BinTableColumnData *> &data)
{
    BinTableHeader header;
    header.version = CURRENT_VERSION;
    uint32_t n_columns = data.size();
    header.n_columns = n_columns;
    if (data.empty())
    {
        header.n_rows = 0;
    }
    else
    {
        header.n_rows = data[0]->size;
    }

    header.columns.reserve(n_columns);
    for (uint32_t i = 0; i < n_columns; i++)
    {
        auto *column_data = data[i];
        header.columns.emplace_back();
        BinTableColumnDefinition &column = header.columns.back();
        column.type = column_data->type;
        column.name = new BinTableString(*(column_data->name));
        column.maxlen = column_data->maxlen;
    }
    return header;
}

template <class T>
struct ptr_less
{
    bool operator()(const T* lhs, const T* rhs) const
    {
        return *lhs < *rhs;
    }
};

BinTableHeader _validate_and_get_header(std::vector<BinTableColumnData *> &data, const std::string &path, bool append)
{
    if (!append)
    {
        return _create_header(data);
    }

    std::ifstream out_file;
    out_file.open(path, std::ios::binary);
    BufferedInputStream stream(out_file, BUFFER_SIZE);

    BinTableHeader old_header(stream);
    BinTableHeader new_header = _create_header(data);

    if (new_header.n_columns != old_header.n_columns)
    {
        throw AppendException("Old and new number of columns mismatch " +
                              std::to_string(old_header.n_columns) + "!=" + std::to_string(new_header.n_columns));
    }

    old_header.n_rows += new_header.n_rows;

    std::map<BinTableString *, uint32_t, ptr_less<BinTableString>> columns_index;

    uint32_t index_for_map = 0;
    for (uint32_t i=0; i<new_header.n_columns; i++)
    {
        auto &col = new_header.columns.data()[i];
        columns_index[col.name] = i;
    }

    std::vector<BinTableColumnData *> ordered_data;
    ordered_data.reserve(old_header.n_columns);

    for (auto it = old_header.columns.begin(); it != old_header.columns.end(); it++)
    {
        auto &old_col = *it;
        auto map_it = columns_index.find(old_col.name);

        if (map_it == columns_index.end())
        {
            throw AppendException("Missing column \"" + old_col.name->to_string() + "\"");
        }

        auto new_col_index = map_it->second;
        auto &new_col = new_header.columns.data()[new_col_index];

        if (new_col.type != old_col.type)
        {
            throw AppendException("Mismatching types for column \"" + old_col.name->to_string() + "\"");
        }

        old_col.maxlen = std::max(old_col.maxlen, new_col.maxlen);

        ordered_data.push_back(data[new_col_index]);
    }

    data.swap(ordered_data);

    return old_header;
}


void _write_rows_block(std::vector<BinTableColumnData *> &data, OutputStream &stream)
{
    uint32_t n_columns = data.size();
    uint64_t n_rows = 0;
    if (n_columns > 0)
    {
        n_rows = data[0]->size;
    };

    FromPythonOperationsSelector selector;
    Writer writer(&selector);
    std::vector<ReadWriteSpecification> cols_specifications;

    stream.write_primitive_endian_aware(n_rows);

    for (uint32_t i = 0; i < n_columns; i++)
    {
        auto column = data[i];

        validate_datatype(column->type);

        auto size = DATATYPE_ELEMENT_SIZE[column->type];

        ReadWriteSpecification spec;
        spec.output_stream = &stream;
        spec.type = column->type;
        spec.maxlen = column->maxlen;
        uint64_t array_length;
        
        if (is_basic_bintable_datatype(column->type))
        {
            array_length = size * n_rows;
        }
        else if (column->type == BINTABLE_UTF32 || column->type == BINTABLE_UTF8)
        {
            array_length = column->maxlen * n_rows;
        }
        else if (column->type == BINTABLE_OBJECT)
        {
            array_length = n_rows * sizeof(PyObject *);
        }

        spec.input_stream = new MemoryInputStream(column->data, array_length);

        cols_specifications.push_back(spec);

        writer.loop(n_rows).write(spec);
    }

    writer.run();

    for (auto spec : cols_specifications) {
        delete spec.input_stream; 
    }
}

void NAMESPACE_BINTABLE::write_table(std::vector<BinTableColumnData *> &data, const std::string &path, bool append)
{
    std::ofstream out_file;
    auto flags = std::ios::out | std::ios::binary;
    if (append)
    {
        flags = flags | std::ios::in;
    }
    else
    {
        flags = flags | std::ios::trunc;
    }

    out_file.open(path, flags);
    BufferedOutputStream stream(out_file, BUFFER_SIZE);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    BinTableHeader header = _validate_and_get_header(data, path, append);
    header.write(stream);
    //Flushing buffer before seek
    stream.flush_buffer();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Writing header = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();

    out_file.seekp(0, std::ios_base::end);
    _write_rows_block(data, stream);

    end = std::chrono::steady_clock::now();
    std::cout << "Writing body = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
}


void _read_rows(BinTableHeader &header, InputStream &stream, std::vector<BinTableColumnData *> &out)
{

    auto n_columns = header.n_columns;
    auto n_rows = header.n_rows;

    
    std::vector<ReadWriteSpecification> cols_specifications;

    for (uint32_t i = 0; i < n_columns; i++)
    {
        BinTableColumnDefinition &column_header = header.columns[i];
        auto type = column_header.type;
        auto maxlen = column_header.maxlen;

        validate_datatype(type);

        auto column_data = new BinTableColumnData();
        column_data->type = type;
        column_data->name = new BinTableString(*(column_header.name));
        column_data->size = header.n_rows;
        column_data->maxlen = maxlen;

        ReadWriteSpecification spec;
        spec.type = type;
        spec.maxlen = maxlen;
        spec.input_stream = &stream;



        auto size = DATATYPE_ELEMENT_SIZE[column_header.type];
        uint64_t array_size;


        if (is_basic_bintable_datatype(column_header.type))
        {
            array_size = n_rows * size;
        }
        else if (column_header.type == BINTABLE_UTF32 || column_header.type == BINTABLE_UTF8)
        {
            array_size = n_rows * maxlen;
        }
        else if (column_header.type == BINTABLE_OBJECT)
        {
            array_size = n_rows*sizeof(PyObject *);
        }

        char *data_array= new char[array_size];

        column_data->data = data_array;
        out.push_back(column_data);

        spec.output_stream = new MemoryOutputStream(data_array, array_size);
        cols_specifications.push_back(spec);
    }

    uint64_t start_block_index = 0;
    while (start_block_index < n_rows)
    {
        uint64_t n_block_rows = 0;
        stream.read_primitive_endian_aware(n_block_rows);

        if (n_block_rows + start_block_index > n_rows)
        {
            throw BinTableException("Too many blocks");
        }

        ToPythonOperationsSelector selector;
        Writer writer(&selector);


        for (auto spec : cols_specifications)
        {
            writer.loop(n_block_rows).write(spec);
        }

        writer.run();

        start_block_index+=n_block_rows;        
    }

    for (auto spec : cols_specifications) {
        delete spec.output_stream; 
    }
}

void NAMESPACE_BINTABLE::read_table(const std::string &path, std::vector<BinTableColumnData *> &out)
{

    std::ifstream in_file;
    in_file.open(path, std::ios::binary);

    BufferedInputStream stream(in_file, BUFFER_SIZE);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    BinTableHeader header(stream);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Reading header = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();

    try
    {
        _read_rows(header, stream, out);
    }
    catch (std::exception ex)
    {
        for (auto col : out)
        {
            delete col->data;
            delete col->name;
            delete col;
        }

        throw;
    }

    end = std::chrono::steady_clock::now();
    std::cout << "Reading body = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
}

BinTableHeader NAMESPACE_BINTABLE::read_header(const std::string &path)
{
    std::ifstream in_file;
    in_file.open(path, std::ios::binary);
    BufferedInputStream stream(in_file, BUFFER_SIZE);

    return BinTableHeader(stream);
}