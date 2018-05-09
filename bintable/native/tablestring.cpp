#include "tablestring.h"
#include "streams/streams.h"
#include <sstream>

using namespace NAMESPACE_BINTABLE;

BinTableString::BinTableString(InputStream &stream)
{
    read(stream);
};

BinTableString::BinTableString(InputStream &stream, uint32_t size)
{
    read(stream, size);
};

BinTableString::BinTableString(const BinTableString &other)
{
    size = other.size;
    buffer_size = size;
    data = new char[size];
    std::copy(other.data, other.data + size, data);
    delete_data = true;
};

BinTableString::BinTableString() {
};

void BinTableString::read_data_array(InputStream &stream)
{
    if (data == nullptr || delete_data == false || size > buffer_size)
    {
        if (delete_data == true)
        {
            delete[] data;
        }
        data = new char[size];
        buffer_size = size;
        delete_data = true;
    }

    stream.read(data, size);
}

void BinTableString::read(InputStream &stream)
{

    stream.read_primitive_endian_aware(size);
    read_data_array(stream);
}

void BinTableString::read(InputStream &stream, uint32_t size)
{
    this->size = size;
    read_data_array(stream);
}

void BinTableString::write(OutputStream &stream)
{
    stream.write_primitive_endian_aware(size);
    stream.write(data, size);
};

void BinTableString::set(char *data, uint32_t size, bool delete_data)
{
    if (this->delete_data)
    {
        delete[] data;
    }
    this->data = data;
    this->size = size;
    this->buffer_size = size;
    this->delete_data = delete_data;
}

char *BinTableString::get_data()
{
    return data;
}

uint32_t BinTableString::get_size()
{
    return size;
}

void BinTableString::read_to_stream(InputStream &input_stream, OutputStream &output_stream, uint32_t &size)
{
    input_stream.read_primitive_endian_aware(size);
    output_stream.write(input_stream, size);
}

void BinTableString::skip(InputStream &stream, uint32_t &size)
{
    stream.read_primitive_endian_aware(size);
    stream.skip(size);
};

std::string BinTableString::to_string() const
{
    std::stringstream stream;
    stream.write(data, size);
    return stream.str();
}

BinTableString::~BinTableString()
{
    if (delete_data)
    {
        delete[] data;
    }
}

bool BinTableString::operator==(const BinTableString &other)
{
    if ((data == nullptr) || (other.data == nullptr))
    {
        if ((data == nullptr) && (other.data == nullptr))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    if (size != other.size)
    {
        return false;
    }

    for (uint32_t i = 0; i < size; i++)
    {
        if (data[i] != other.data[i])
        {
            return false;
        }
    }

    return true;
}

bool BinTableString::operator!=(const BinTableString &other)
{
    return !((*this) == other);
}

bool BinTableString::operator<(const BinTableString &other) const
{
    if (size != other.size)
    {
        return size < other.size;
    }

    for (uint32_t i = 0; i < size; i++)
    {
        if (data[i] != other.data[i])
        {
            return data[i] < other.data[i];
        }
    }

    return false;
}