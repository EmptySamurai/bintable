#include "streams/memorystreams.h"

using namespace NAMESPACE_BINTABLE;

MemoryInputStream::MemoryInputStream(char *input_data, std::streamsize input_data_size) : input_data(input_data),
                                                                                          input_data_size(input_data_size),
                                                                                          current_position(0)
{
}

void MemoryInputStream::read(char *data, const std::streamsize size)
{
    if (size > input_data_size - current_position)
    {
        throw_not_enough_bytes();
    }

    std::copy(input_data + current_position, input_data + current_position + size, data);
    current_position += size;
}

void MemoryInputStream::skip(const std::streamsize size)
{
    if (size > input_data_size - current_position)
    {
        throw_not_enough_bytes();
    }

    current_position += size;
}

MemoryOutputStream::MemoryOutputStream(char *output_data, std::streamsize output_data_max_size) : output_data(output_data),
                                                                                                  output_data_max_size(output_data_max_size),
                                                                                                  current_position(0)
{
}

void MemoryOutputStream::write(const char *data, const std::streamsize size)
{
    if (size > output_data_max_size - current_position)
    {
        throw StreamFinishedException("Not enough bytes to write in memory");
    }

    std::copy(data, data + size, output_data + current_position);
    current_position += size;
}

void MemoryOutputStream::write(InputStream &stream, const std::streamsize size)
{
    if (size > output_data_max_size - current_position)
    {
        throw StreamFinishedException("Not enough bytes to write in memory");
    }
    stream.read(output_data + current_position, size);
    current_position += size;
    
}