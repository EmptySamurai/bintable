#pragma once
#include "common.h"
#include "streams/streams.h"
#include <iostream>

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)


class BufferedInputStream : public InputStream
{

  public:
    BufferedInputStream(std::istream& stream, std::streamsize buffer_max_size);
    void read(char *data, const std::streamsize size) override;
    void skip(const std::streamsize size) override;
    ~BufferedInputStream() override;

  private:
    std::istream* stream;
    const std::streamsize buffer_max_size;
    char *buffer;
    std::streamsize buffer_size;
    std::streamsize current_buffer_position;
    void fill_empty_buffer();
};

class BufferedOutputStream : public OutputStream
{

  public:
    BufferedOutputStream(std::ostream& stream, std::streamsize buffer_max_size);
    void flush_buffer();
    void write(const char *data, const std::streamsize size) override;
    void write(InputStream& stream, const std::streamsize size) override;
    ~BufferedOutputStream() override;

  private:
    std::ostream* stream;
    const std::streamsize buffer_max_size;
    char *buffer;
    std::streamsize buffer_size;
};


NAMESPACE_END(NAMESPACE_BINTABLE)
