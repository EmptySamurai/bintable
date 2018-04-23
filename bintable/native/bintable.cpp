#include "bintable.h"
#include "pythonutils.h"

#include "ioutils.h"
#include "Python.h"
#include <ostream>
#include <fstream>
#include <functional>
#include <map>
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
    for (auto i = 0; i < n_columns; i++)
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
    bool operator()(T *lhs, T *rhs)
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

    std::map<BinTableString *, BinTableColumnDefinition *, ptr_less<BinTableString>> columns_index;

    for (auto it = new_header.columns.begin(); it != new_header.columns.end(); it++)
    {
        auto col = &(*it);
        columns_index[col->name] = col;
    }

    for (auto it = old_header.columns.begin(); it != old_header.columns.end(); it++)
    {
        auto &old_col = *it;
        auto map_it = columns_index.find(old_col.name);

        if (map_it == columns_index.end())
        {
            throw AppendException("Missing column \"" + old_col.name->to_string() + "\"");
        }

        auto new_col = map_it->second;

        if (new_col->type != old_col.type)
        {
            throw AppendException("Mismatching types for column \"" + old_col.name->to_string() + "\"");
        }

        old_col.maxlen = std::max(old_col.maxlen, new_col->maxlen);
    }

    return old_header;
}

inline void _write_row_fixed_length_string(BufferedOutputStream &stream, char *data, uint64_t &index, uint8_t size, uint32_t maxlen)
{
    char *data_start = data + index * maxlen;
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
    temp_string.write(stream);
}

inline void _write_row_object(BufferedOutputStream &stream, PyObject **data, uint64_t &index)
{
    BinTableString temp_string;
    python_object_to_table_string(data[index], temp_string);
    temp_string.write(stream);
}

void _write_rows_block(std::vector<BinTableColumnData *> &data, BufferedOutputStream &stream)
{
    uint32_t n_columns = data.size();
    uint64_t n_rows = 0;
    if (n_columns > 0)
    {
        n_rows = data[0]->size;
    };

    stream.write_primitive(n_rows);

    for (uint32_t i = 0; i < n_columns; i++)
    {
        auto column = data[i];
        auto data_array = data[i]->data;

        auto size = DATATYPE_ELEMENT_SIZE[column->type];

        if (is_basic_bintable_datatype(column->type))
        {
            stream.write(data_array, size * n_rows);
        }
        else if (column->type == BINTABLE_UTF32 || column->type == BINTABLE_UTF8)
        {
            auto maxlen = column->maxlen;
            for (uint64_t i = 0; i < n_rows; i++)
            {
                _write_row_fixed_length_string(stream, data_array, i, size, maxlen);
            }
        }
        else if (column->type == BINTABLE_OBJECT)
        {
            PyObject **objects_array = reinterpret_cast<PyObject **>(data_array);
            for (uint64_t i = 0; i < n_rows; i++)
            {
                _write_row_object(stream, objects_array, i);
            }
        }
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


inline void _read_row_fixed_string(BufferedInputStream &stream, char *data, const uint64_t &index, uint32_t maxlen)
{
    uint32_t size = 0;
    BinTableString::read_to_buffer(stream, data + maxlen * index, size);
}

inline void _read_row_object(BufferedInputStream &stream, PyObject **data, const uint64_t &index)
{
    BinTableString table_string(stream);
    data[index] = table_string_to_python_object(table_string);
}

void _read_rows(BinTableHeader &header, BufferedInputStream &stream, std::vector<BinTableColumnData *> &out)
{

    auto n_columns = header.n_columns;
    auto n_rows = header.n_rows;

    for (uint32_t i = 0; i < n_columns; i++)
    {
        BinTableColumnDefinition &column_header = header.columns[i];
        auto column_data = new BinTableColumnData();

        auto type = column_header.type;
        column_data->type = type;
        column_data->name = new BinTableString(*(column_header.name));
        column_data->size = header.n_rows;
        auto maxlen = column_header.maxlen;
        column_data->maxlen = maxlen;

        auto size = DATATYPE_ELEMENT_SIZE[column_header.type];

        char *data_array=nullptr;
        if (is_basic_bintable_datatype(column_header.type))
        {
            data_array = new char[n_rows * size];
        }
        else if (column_header.type == BINTABLE_UTF32 || column_header.type == BINTABLE_UTF8)
        {
            auto data_array_size = n_rows * maxlen;
            data_array = new char[data_array_size];
            std::fill(data_array, data_array + data_array_size, 0);
        }
        else if (column_header.type == BINTABLE_OBJECT)
        {
            PyObject **objects_array = new PyObject *[n_rows];
            data_array = reinterpret_cast<char *>(objects_array);
        }

        column_data->data = data_array;
        out.push_back(column_data);
    }

    uint64_t start_row_index = 0;
    while (start_row_index < n_rows)
    {
        uint64_t n_block_rows = 0;
        stream.read_primitive(n_block_rows);

        if (n_block_rows + start_row_index > n_rows)
        {
            throw BinTableException("Too many blocks");
        }


        for (auto it = out.begin(); it != out.end(); it++)
        {
            auto col = *it;
            auto data_array = col->data;
            auto size = DATATYPE_ELEMENT_SIZE[col->type];

            if (is_basic_bintable_datatype(col->type))
            {
                stream.read(data_array + start_row_index * size, size * n_block_rows);
            }
            else if (col->type == BINTABLE_UTF32 || col->type == BINTABLE_UTF8)
            {
                for (uint64_t i = 0; i < n_block_rows; i++)
                {
                    _read_row_fixed_string(stream, data_array, i + start_row_index, col->maxlen);
                }
            }
            else if (col->type == BINTABLE_OBJECT)
            {
                PyObject **objects_array = reinterpret_cast<PyObject **>(data_array);
                for (uint64_t i = 0; i < n_block_rows; i++)
                {
                    _read_row_object(stream, objects_array, i + start_row_index);
                }
            }
        }

        start_row_index += n_block_rows; 
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

        throw ex;
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