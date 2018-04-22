#include "header.h"
#include "ioutils.h"
#include <iostream>
#include <fstream>
#include <iostream>

using namespace NAMESPACE_BINTABLE;


BinTableColumnDefinition::BinTableColumnDefinition(BufferedInputStream& stream) {
    stream.read_primitive(type);
    name = new BinTableString(stream);
    if (has_maxlen()) {
        stream.read_primitive(maxlen);
    } else {
        maxlen = DATATYPE_ELEMENT_SIZE[type];
    }
};

BinTableColumnDefinition::BinTableColumnDefinition() = default;

void BinTableColumnDefinition::write(BufferedOutputStream& stream) {
    stream.write_primitive(type);
    name->write(stream);
    if (has_maxlen()) {
        stream.write_primitive(maxlen);
    }
};

BinTableColumnDefinition::~BinTableColumnDefinition() {
    delete name;
}

bool BinTableColumnDefinition::has_maxlen() {
    return (type == BINTABLE_UTF32) || (type == BINTABLE_UTF8);
} 



BinTableHeader::BinTableHeader(BufferedInputStream& stream) {
    stream.read_primitive(version);
    stream.read_primitive(n_rows);
    stream.read_primitive(n_columns);

    columns = new std::vector<BinTableColumnDefinition*>();
    columns->reserve(n_columns);
    for (auto i=0; i< n_columns; i++) {
        columns->push_back(new BinTableColumnDefinition(stream));
    }
};

BinTableHeader::BinTableHeader() = default;

void BinTableHeader::write(BufferedOutputStream& stream) {
    stream.write_primitive(version);
    stream.write_primitive(n_rows);
    stream.write_primitive(n_columns);

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
