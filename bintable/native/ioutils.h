#pragma once
#include "common.h"
#include <istream>
#include <iostream>
#include "exceptions.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class BufferedInputStream
{

  public:
    BufferedInputStream(std::istream& stream, std::size_t buffer_max_size) : buffer_max_size(buffer_max_size)
    {
        this->stream = &stream;
        buffer = new char[buffer_max_size];
        buffer_size = 0;
        current_buffer_position = 0;
    }
    void read(char *data, std::streamsize size)
    {
        std::streamsize bytesLeftInBuffer = buffer_size - current_buffer_position;
        if (bytesLeftInBuffer < size)
        {
            std::copy(buffer + current_buffer_position, buffer + buffer_size, data);
            std::streamsize bytesLeftNotWritten = size - bytesLeftInBuffer;
            stream->read(data + bytesLeftInBuffer, bytesLeftNotWritten);

            if (bytesLeftNotWritten != stream->gcount())
            {
                throw StreamFinishedException("Not enough bytes to read");
            }

            stream->read(buffer, buffer_max_size);
            current_buffer_position = 0;
            buffer_size = stream->gcount();
        }
        else
        {
            std::copy(buffer + current_buffer_position, buffer + current_buffer_position + size, data);
            current_buffer_position += size;
        }
    };

    template <class T>
    void read_primitive(T &val)
    {
        read(reinterpret_cast<char *>(&val), sizeof(val));
    }
    ~BufferedInputStream() {
        delete[] buffer;
    }

  private:
    std::istream* stream;
    std::size_t buffer_max_size;
    char *buffer;
    std::streamsize buffer_size;
    std::streamsize current_buffer_position;
};

class BufferedOutputStream
{

  public:
    BufferedOutputStream(std::ostream& stream, std::size_t buffer_max_size)  : buffer_max_size(buffer_max_size)
    {
        this->stream = &stream;
        buffer = new char[buffer_max_size];
        buffer_size = 0;
    }

    void flush_buffer()
    {
        stream->write(buffer, buffer_size);
        buffer_size = 0;
    };

    void write(char *data, std::streamsize size)
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

    template <class T>
    void write_primitive(T &val)
    {
        write(reinterpret_cast<char *>(&val), sizeof(val));
    }

    ~BufferedOutputStream() {
        delete[] buffer;
    }

  private:
    std::ostream* stream;
    std::size_t buffer_max_size;
    char *buffer;
    std::streamsize buffer_size;
};

NAMESPACE_END(NAMESPACE_BINTABLE)
