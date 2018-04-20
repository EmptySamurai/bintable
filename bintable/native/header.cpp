#include "header.h"
#include "utils.h"
#include <iostream>
#include <fstream>

using namespace NAMESPACE_BINTABLE;

BinTableColumnDefinition::BinTableColumnDefinition(std::istream& stream) {
    read_from_stream(stream, type);
    name = new BinTableString(stream);
};

BinTableColumnDefinition::BinTableColumnDefinition() = default;

void BinTableColumnDefinition::write(std::ostream& stream) {
    write_to_stream(stream, type);
    name->write(stream);
};

BinTableColumnDefinition::~BinTableColumnDefinition() {
    delete name;
}


BinTableStringColumnDefinition::BinTableStringColumnDefinition(std::istream& stream) : BinTableColumnDefinition(stream) {
    read_from_stream(stream, maxlen);
};

BinTableStringColumnDefinition::BinTableStringColumnDefinition() = default;

void BinTableStringColumnDefinition::write(std::ostream& stream) {
    BinTableColumnDefinition::write(stream);
    write_to_stream(stream, maxlen);
};



BinTableHeader::BinTableHeader(std::istream& stream) {
    read_from_stream(stream, version);
    read_from_stream(stream, n_rows);
    read_from_stream(stream, n_columns);

    columns = new std::vector<BinTableColumnDefinition*>();
    columns->reserve(n_columns);
    for (auto i=0; i< n_columns; i++) {
        columns->push_back(new BinTableColumnDefinition(stream));
    }
};

BinTableHeader::BinTableHeader() = default;

void BinTableHeader::write(std::ostream& stream) {
    write_to_stream(stream, version);
    write_to_stream(stream, n_rows);
    write_to_stream(stream, n_columns);

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
