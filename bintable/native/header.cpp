#include "header.h"
#include "ioutils.h"
#include "exceptions.h"

using namespace NAMESPACE_BINTABLE;

BinTableColumnDefinition::BinTableColumnDefinition(BufferedInputStream &stream)
{
    stream.read_primitive(type);
    if (!is_valid_datatype(type))
    {
        throw UnknownDatatypeException("Unknown datatype");
    }
    name = new BinTableString(stream);
    if (has_maxlen())
    {
        stream.read_primitive(maxlen);
    }
    else
    {
        maxlen = DATATYPE_ELEMENT_SIZE[type];
    }
};

BinTableColumnDefinition::BinTableColumnDefinition() = default;

BinTableColumnDefinition::BinTableColumnDefinition(const BinTableColumnDefinition &other)
{
    type = other.type;
    name = other.name;
    if (other.name != nullptr)
    {
        name = new BinTableString(*(other.name));
    }
}

void BinTableColumnDefinition::write(BufferedOutputStream &stream) const
{
    stream.write_primitive(type);
    name->write(stream);
    if (has_maxlen())
    {
        stream.write_primitive(maxlen);
    }
};

BinTableColumnDefinition::~BinTableColumnDefinition()
{
    delete name;
}

bool BinTableColumnDefinition::has_maxlen() const
{
    return (type == BINTABLE_UTF32) || (type == BINTABLE_UTF8);
}

BinTableHeader::BinTableHeader(BufferedInputStream &stream)
{
    stream.read_primitive(version);
    stream.read_primitive(n_rows);
    stream.read_primitive(n_columns);

    columns.reserve(n_columns);

    for (uint32_t i = 0; i < n_columns; i++)
    {
        columns.emplace_back(stream);
    }
};

BinTableHeader::BinTableHeader() = default;

BinTableHeader::BinTableHeader(const BinTableHeader &other) = default;

void BinTableHeader::write(BufferedOutputStream &stream) const
{
    stream.write_primitive(version);
    stream.write_primitive(n_rows);
    stream.write_primitive(n_columns);

    for (auto it = columns.begin(); it!=columns.end(); it++)
    {
        (*it).write(stream);
    }
};

