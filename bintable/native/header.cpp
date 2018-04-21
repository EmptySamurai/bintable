#include "header.h"
#include "ioutils.h"
#include <iostream>
#include <fstream>
#include <iostream>

using namespace NAMESPACE_BINTABLE;


BinTableColumnDefinition::BinTableColumnDefinition(std::istream& stream) {
    read_primitive_from_stream(stream, type);
    name = new BinTableString(stream);
    if (has_maxlen()) {
        read_primitive_from_stream(stream, maxlen);
    } else {
        maxlen = DATATYPE_ELEMENT_SIZE[type];
    }
};

BinTableColumnDefinition::BinTableColumnDefinition() = default;

void BinTableColumnDefinition::write(std::ostream& stream) {
    write_primitive_to_stream(stream, type);
    name->write(stream);
    if (has_maxlen()) {
        write_primitive_to_stream(stream, maxlen);
    }
};

BinTableColumnDefinition::~BinTableColumnDefinition() {
    delete name;
}

bool BinTableColumnDefinition::has_maxlen() {
    return (type == BINTABLE_UTF32) || (type == BINTABLE_UTF8);
} 



BinTableHeader::BinTableHeader(std::istream& stream) {
    read_primitive_from_stream(stream, version);
    read_primitive_from_stream(stream, n_rows);
    read_primitive_from_stream(stream, n_columns);

    columns = new std::vector<BinTableColumnDefinition*>();
    columns->reserve(n_columns);
    for (auto i=0; i< n_columns; i++) {
        columns->push_back(new BinTableColumnDefinition(stream));
    }
};

BinTableHeader::BinTableHeader() = default;

void BinTableHeader::write(std::ostream& stream) {
    write_primitive_to_stream(stream, version);
    write_primitive_to_stream(stream, n_rows);
    write_primitive_to_stream(stream, n_columns);

    for (auto i=0; i< n_columns; i++) {
        (*columns)[i]->write(stream);
    }
};

BinTableHeader::~BinTableHeader() {
    if (columns != nullptr) {
        for (auto &column : *columns) {
            delete column;
        }
    }
    delete columns;
};
