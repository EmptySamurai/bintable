#include "bintable.h"
#include <ostream>
#include <fstream>
#include <functional>

using namespace NAMESPACE_BINTABLE;

BinTableHeader *_create_header(std::vector<BinTableColumnData *> &data) {
    auto *header = new BinTableHeader();
    header->version = CURRENT_VERSION;
    uint32_t n_columns = data.size();
    header->n_columns = n_columns;
    if (data.empty()) {
        header->n_rows = 0;
    } else {
        header->n_rows = data[0]->size;
    }

    header->columns = new std::vector<BinTableColumnDefinition *>();
    header->columns->reserve(n_columns);
    for (auto i = 0; i < n_columns; i++) {
        auto *columnData = data[i];
        auto *column = new BinTableColumnDefinition();
        column->type = columnData->type;
        column->name = columnData->name;
        header->columns->push_back(column);
    }
    return header;
}


void _write_row_basic_value(std::ostream &stream, char* data, uint64_t& index, uint8_t size) {
    char* data_start = data+index*size;
    stream.write(data_start, size);
}

BinTableString _temp_string;
void _write_row_utf32_value(std::ostream &stream, char* data, uint64_t& index, uint32_t maxlen) {
    const uint8_t symbol_size = 4;
    char* data_start = data+index*symbol_size*maxlen;
    char* data_end = data_start+maxlen*symbol_size;
    uint32_t len = maxlen;
    for (uint32_t i=0; i<maxlen; i++) {
        if (data_end[-1] || data_end[-2] || data_end[-3] || data_end[-4]) {
            break;
        } else {
            len--;
            data_end -= symbol_size;
        }
    }
    _temp_string.data=data_start;
    _temp_string.size = len*symbol_size;
    _temp_string.write(stream);
}

typedef void (*writerfunc)(uint64_t&);


void _write_rows(std::vector<BinTableColumnData *> &data, std::ostream &stream) {
    auto n_columns = data.size();
    auto n_rows = 0;
    if (n_columns>0) {
        n_rows=data[0]->size;
    };
    std::vector<std::function<void(uint64_t&)>> funcs;
    for (uint64_t i = 0; i < n_columns; i++) {
        auto column = data[i];
        auto dataArray = data[i]->data;
        if (is_basic_bintable_datatype(column->type)) {
            auto size = BASIC_DATATYPES_SIZE[column->type];
            funcs.push_back([&stream,  dataArray, size](uint64_t& index) {
                _write_row_basic_value(stream, dataArray, index, size);
            });
        } else if (column->type == BINTABLE_UTF32){
            auto maxlen = column->maxlen;
            funcs.push_back([&stream, dataArray, maxlen](uint64_t& index) {
                _write_row_utf32_value(stream, dataArray, index, maxlen);
            });
        }
    }

    for (uint64_t i = 0; i < n_rows; i++) {
        for (auto j = 0; j < n_columns; j++) {
            funcs[j](i);
        }
    }
}

void NAMESPACE_BINTABLE::write_table(std::vector<BinTableColumnData *> &data, const std::string &path, bool append) {
    BinTableHeader *header;
    std::ofstream outFile;

    outFile.open(path, std::ios::binary);
    header = _create_header(data);
    header->write(outFile);

    _write_rows(data, outFile);

    delete header;
}