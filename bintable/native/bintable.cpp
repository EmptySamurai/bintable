#include "bintable.h"
#include "pythonutils.h"

#include "ioutils.h"
#include "Python.h"
#include <ostream>
#include <fstream>
#include <functional>
#include <chrono>
#include <iostream>

using namespace NAMESPACE_BINTABLE;

const int BUFFER_SIZE = 100000;

BinTableHeader *_create_header(std::vector<BinTableColumnData *> &data)
{
    auto *header = new BinTableHeader();
    header->version = CURRENT_VERSION;
    uint32_t n_columns = data.size();
    header->n_columns = n_columns;
    if (data.empty())
    {
        header->n_rows = 0;
    }
    else
    {
        header->n_rows = data[0]->size;
    }

    header->columns = new std::vector<BinTableColumnDefinition *>();
    header->columns->reserve(n_columns);
    for (auto i = 0; i < n_columns; i++)
    {
        auto *column_data = data[i];
        auto type = column_data->type;
        BinTableColumnDefinition *column = new BinTableColumnDefinition();
        column->type = column_data->type;
        column->name = column_data->name->copy();
        column->maxlen = column_data->maxlen;
        header->columns->push_back(column);
    }
    return header;
}

inline void _write_row_basic_value(BufferedOutputStream &stream, char *data, uint64_t &index, uint8_t size)
{
    char *data_start = data + index * size;
    stream.write(data_start, size);
};

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

void _write_rows(std::vector<BinTableColumnData *> &data, BufferedOutputStream &stream)
{
    auto n_columns = data.size();
    auto n_rows = 0;
    if (n_columns > 0)
    {
        n_rows = data[0]->size;
    };

    std::vector<std::function<void(uint64_t &)>> funcs;
    for (uint32_t i = 0; i < n_columns; i++)
    {
        auto column = data[i];
        auto data_array = data[i]->data;

        auto size = DATATYPE_ELEMENT_SIZE[column->type];

        if (is_basic_bintable_datatype(column->type))
        {
            funcs.push_back([&stream, data_array, size](uint64_t &index) {
                _write_row_basic_value(stream, data_array, index, size);
            });
        }
        else if (column->type == BINTABLE_UTF32 || column->type == BINTABLE_UTF8)
        {
            auto maxlen = column->maxlen;
            funcs.push_back([&stream, data_array, size, maxlen](uint64_t &index) {
                _write_row_fixed_length_string(stream, data_array, index, size, maxlen);
            });
        }
        else if (column->type == BINTABLE_OBJECT)
        {
            PyObject **objects_array = reinterpret_cast<PyObject **>(data_array);
            funcs.push_back([&stream, objects_array](uint64_t &index) {
                _write_row_object(stream, objects_array, index);
            });
        }
    }

    for (uint64_t i = 0; i < n_rows; i++)
    {
        for (uint32_t j = 0; j < n_columns; j++)
        {
            funcs[j](i);
        }
    }
}

void NAMESPACE_BINTABLE::write_table(std::vector<BinTableColumnData *> &data, const std::string &path, bool append)
{
    BinTableHeader *header;
    std::ofstream out_file;

    BufferedOutputStream stream(out_file, BUFFER_SIZE);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    out_file.open(path, std::ios::binary);

    header = _create_header(data);
    header->write(stream);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Writing header = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();

    try
    {
        _write_rows(data, stream);
    }
    catch (std::exception ex)
    {
        out_file.close();
        delete header;
        throw ex;
    }
    
    stream.flush_buffer();

    end = std::chrono::steady_clock::now();
    std::cout << "Writing body = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    out_file.close();

    delete header;
}

inline void _read_row_basic_value(BufferedInputStream &stream, char *data, uint64_t &index, uint8_t size)
{
    stream.read(data + index * size, size);
};

inline void _read_row_fixed_string(BufferedInputStream &stream, char *data, uint64_t &index, uint32_t maxlen)
{
    uint32_t size = 0;
    BinTableString::read_to_buffer(stream, data + maxlen * index, size);
}

inline void _read_row_object(BufferedInputStream &stream, PyObject **data, uint64_t &index)
{
    BinTableString table_string(stream);
    data[index] = table_string_to_python_object(table_string);
}

void _read_rows(BinTableHeader *header, BufferedInputStream &stream, std::vector<BinTableColumnData *> &out)
{

    auto n_columns = header->n_columns;
    auto n_rows = header->n_rows;

    std::vector<std::function<void(uint64_t &)>> funcs;
    for (uint32_t i = 0; i < n_columns; i++)
    {
        auto column_header = (*(header->columns))[i];
        auto column_data = new BinTableColumnData();

        auto type = column_header->type;
        column_data->type = type;
        column_data->name = column_header->name->copy();
        column_data->size = header->n_rows;
        auto maxlen = column_header->maxlen;
        column_data->maxlen = maxlen;

        auto size = DATATYPE_ELEMENT_SIZE[column_header->type];

        char *data_array;
        if (is_basic_bintable_datatype(column_header->type))
        {
            data_array = new char[n_rows * size];
            funcs.push_back([&stream, data_array, size](uint64_t &index) {
                _read_row_basic_value(stream, data_array, index, size);
            });
        }
        else if (column_header->type == BINTABLE_UTF32 || column_header->type == BINTABLE_UTF8)
        {
            auto data_array_size = n_rows * maxlen;
            data_array = new char[data_array_size];
            std::fill(data_array, data_array + data_array_size, 0);
            funcs.push_back([&stream, data_array, maxlen](uint64_t &index) {
                _read_row_fixed_string(stream, data_array, index, maxlen);
            });
        }
        else if (column_header->type == BINTABLE_OBJECT)
        {
            PyObject **objects_array = new PyObject *[n_rows];
            data_array = reinterpret_cast<char *>(objects_array);
            funcs.push_back([&stream, objects_array](uint64_t &index) {
                _read_row_object(stream, objects_array, index);
            });
        }

        column_data->data = data_array;
        out.push_back(column_data);
    }

    for (uint64_t i = 0; i < n_rows; i++)
    {
        for (uint32_t j = 0; j < n_columns; j++)
        {
            funcs[j](i);
        }
    }
}

void NAMESPACE_BINTABLE::read_table(const std::string &path, std::vector<BinTableColumnData *> &out)
{

    std::ifstream in_file;
    in_file.open(path, std::ios::binary);

    BufferedInputStream stream(in_file, BUFFER_SIZE);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    BinTableHeader *header = new BinTableHeader(stream);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Reading header = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();

    try
    {
        _read_rows(header, stream, out);
    }
    catch (std::exception ex)
    {
        delete header;
        for (auto col : out)
        {
            delete col->data;
            delete col->name;
            delete col;
        }
        out.clear();

        in_file.close();

        throw ex;
    }

    end = std::chrono::steady_clock::now();
    std::cout << "Reading body = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    delete header;
}