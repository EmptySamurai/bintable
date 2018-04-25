#pragma once
#include <cstdint>
#include <string>
#include "common.h"
#include "streams/streams.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class BinTableString
{
  public:
    //In bytes
    uint32_t size = 0;
    char *data = nullptr;
    bool delete_data = false;

    BinTableString();
    BinTableString(const BinTableString& other);
    explicit BinTableString(InputStream &stream);
    BinTableString(InputStream &stream, uint32_t size);
    void write(OutputStream &stream);
    static void skip(InputStream &stream, uint32_t &size);
    static void read_to_buffer(InputStream &stream, char *buffer, uint32_t &size);
    static void read_to_stream(InputStream &input_stream, OutputStream &output_stream, uint32_t &size);

    std::string to_string() const;
    ~BinTableString();

    bool operator==(const BinTableString &other);
    bool operator!=(const BinTableString &other);
    bool operator<(const BinTableString &other) const;

  private:
    void read_data_array(InputStream &stream);
};



NAMESPACE_END(NAMESPACE_BINTABLE)
