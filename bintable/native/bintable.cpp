#include "bintable.h"
#include "ioutils.h"
#include <ostream>
#include <fstream>
#include <functional>
#include <chrono>
#include <iostream>

using namespace NAMESPACE_BINTABLE;

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
        auto *columnData = data[i];
        auto type = columnData->type;
        BinTableColumnDefinition *column = new BinTableColumnDefinition();
        column->type = columnData->type;
        column->name = columnData->name->copy();
        column->maxlen = columnData->maxlen;
        header->columns->push_back(column);
    }
    return header;
}

inline void _write_row_basic_value(std::ostream &stream, char *data, uint64_t &index, uint8_t size)
{
    char *data_start = data + index * size;
    write_to_stream_buffered(stream, data_start, size);
};

BinTableString _temp_string;
inline void _write_row_fixed_length_string(std::ostream &stream, char *data, uint64_t &index, uint8_t size, uint32_t maxlen)
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
    _temp_string.data = data_start;
    _temp_string.size = len;
    _temp_string.write(stream);
}

void _write_rows(std::vector<BinTableColumnData *> &data, std::ostream &stream)
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
        auto dataArray = data[i]->data;

        auto size = DATATYPE_ELEMENT_SIZE[column->type];

        if (is_basic_bintable_datatype(column->type))
        {
            funcs.push_back([&stream, dataArray, size](uint64_t &index) {
                _write_row_basic_value(stream, dataArray, index, size);
            });
        }
        else if (column->type == BINTABLE_UTF32 || column->type == BINTABLE_UTF8)
        {
            auto maxlen = column->maxlen;
            funcs.push_back([&stream, dataArray, size, maxlen](uint64_t &index) {
                _write_row_fixed_length_string(stream, dataArray, index, size, maxlen);
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
    clear_buffer();

    BinTableHeader *header;
    std::ofstream outFile;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    outFile.open(path, std::ios::binary);

    header = _create_header(data);
    header->write(outFile);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Writing header = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();

    _write_rows(data, outFile);
    flush_buffer(outFile);

    end = std::chrono::steady_clock::now();
    std::cout << "Writing body = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    outFile.close();

    delete header;
}

inline void _read_row_basic_value(std::istream &stream, char *data, uint64_t &index, uint8_t size)
{
    read_from_stream_buffered(stream, data + index * size, size);
};

inline void _read_row_fixed_string(std::istream &stream, char *data, uint64_t &index, uint32_t maxlen)
{
    uint32_t size = 0;
    BinTableString::read_to_buffer(stream, data + maxlen * index, size);
}

void _read_rows(BinTableHeader *header, std::istream &stream, std::vector<BinTableColumnData *> &out)
{

    auto n_columns = header->n_columns;
    auto n_rows = header->n_rows;

    std::vector<std::function<void(uint64_t &)>> funcs;
    for (uint32_t i = 0; i < n_columns; i++)
    {
        auto columnHeader = (*(header->columns))[i];
        auto columnData = new BinTableColumnData();

        auto type = columnHeader->type;
        columnData->type = type;
        columnData->name = columnHeader->name->copy();
        columnData->size = header->n_rows;
        auto maxlen = columnHeader->maxlen;
        columnData->maxlen = maxlen;

        auto size = DATATYPE_ELEMENT_SIZE[columnHeader->type];

        char *dataArray;
        if (is_basic_bintable_datatype(columnHeader->type))
        {
            dataArray = new char[n_rows * size];
            funcs.push_back([&stream, dataArray, size](uint64_t &index) {
                _read_row_basic_value(stream, dataArray, index, size);
            });
        }
        else if (columnHeader->type == BINTABLE_UTF32 || columnHeader->type == BINTABLE_UTF8)
        {
            auto dataArraySize = n_rows * maxlen;
            dataArray = new char[dataArraySize];
            std::fill(dataArray, dataArray+dataArraySize, 0);
            funcs.push_back([&stream, dataArray, maxlen](uint64_t &index) {
                _read_row_fixed_string(stream, dataArray, index, maxlen);
            });
        }

        columnData->data = dataArray;
        out.push_back(columnData);
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
    clear_buffer();

    std::ifstream inFile;
    inFile.open(path, std::ios::binary);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    BinTableHeader *header = new BinTableHeader(inFile);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Reading header = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    begin = std::chrono::steady_clock::now();

    _read_rows(header, inFile, out);

    end = std::chrono::steady_clock::now();
    std::cout << "Reading body = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    delete header;
}