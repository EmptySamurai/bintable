#include "streams/bufferedstreams.h"

using namespace NAMESPACE_BINTABLE;

BufferedInputStream::BufferedInputStream(std::istream &stream, std::streamsize buffer_max_size) : buffer_max_size(buffer_max_size)
{
    this->stream = &stream;
    buffer = new char[buffer_max_size];
    buffer_size = 0;
    current_buffer_position = 0;
}

void BufferedInputStream::read(char *data, const std::streamsize size)
{
    std::streamsize bytes_left_in_buffer = buffer_size - current_buffer_position;
    if (bytes_left_in_buffer < size)
    {
        std::copy(buffer + current_buffer_position, buffer + buffer_size, data);
        std::streamsize bytesLeftNotWritten = size - bytes_left_in_buffer;
        stream->read(data + bytes_left_in_buffer, bytesLeftNotWritten);

        if (bytesLeftNotWritten != stream->gcount())
        {
            throw_not_enough_bytes();
        }

        fill_empty_buffer();
    }
    else
    {
        std::copy(buffer + current_buffer_position, buffer + current_buffer_position + size, data);
        current_buffer_position += size;
    }
};

void BufferedInputStream::skip(const std::streamsize size) {
    std::streamsize bytes_left_in_buffer = buffer_size - current_buffer_position;
    if (bytes_left_in_buffer < size)
    {
        auto bytes_to_ignore = size - bytes_left_in_buffer; 
        stream->ignore(bytes_to_ignore);

        if (bytes_to_ignore != stream->gcount())
        {
            throw_not_enough_bytes();
        }

        fill_empty_buffer();
    } else {
        current_buffer_position += size;
    }
}

void BufferedInputStream::fill_empty_buffer() {
    stream->read(buffer, buffer_max_size);
    current_buffer_position = 0;
    buffer_size = stream->gcount();
}

BufferedInputStream::~BufferedInputStream()
{
    delete[] buffer;
}

BufferedOutputStream::BufferedOutputStream(std::ostream &stream, std::streamsize buffer_max_size) : buffer_max_size(buffer_max_size)
{
    this->stream = &stream;
    buffer = new char[buffer_max_size];
    buffer_size = 0;
}

void BufferedOutputStream::flush_buffer()
{
    stream->write(buffer, buffer_size);
    buffer_size = 0;
};

void BufferedOutputStream::write(const char *data, const std::streamsize size)
{
    if (buffer_size + size > buffer_max_size)
    {
        flush_buffer();
        stream->write(data, size);
    }
    else
    {
        std::copy(data, data + size, buffer + buffer_size);
        buffer_size += size;
    }
};

void BufferedOutputStream::write(InputStream &input_stream, const std::streamsize size)
{
    if (buffer_size + size > buffer_max_size)
    {
        flush_buffer();
        auto left_bytes_to_read = size;
        while (left_bytes_to_read > 0)
        {
            auto bytes_read = std::min(left_bytes_to_read, buffer_max_size);
            input_stream.read(buffer, bytes_read);
            stream->write(buffer, bytes_read);
            left_bytes_to_read -= bytes_read;
        }       
    }
    else
    {
        input_stream.read(buffer + buffer_size, size);
        buffer_size += size;
    }
}

BufferedOutputStream::~BufferedOutputStream()
{
    flush_buffer();
    delete[] buffer;
}
