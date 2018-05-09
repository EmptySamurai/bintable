#pragma once
#include <cstdint>
#include <string>
#include "common.h"
#include "streams/streams.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class BinTableString
{
  public:

    BinTableString();
    BinTableString(const BinTableString& other);
    explicit BinTableString(InputStream &stream);
    BinTableString(InputStream &stream, uint32_t size);

    void read(InputStream &stream);
    void read(InputStream &stream, uint32_t size);
    void write(OutputStream &stream);
    void set(char* data, uint32_t size, bool delete_data);

    char* get_data();
    uint32_t get_size();

    static void read_to_stream(InputStream &input_stream, OutputStream &output_stream, uint32_t &size);
    static void skip(InputStream &stream, uint32_t &size);

    std::string to_string() const;
    ~BinTableString();

    bool operator==(const BinTableString &other);
    bool operator!=(const BinTableString &other);
    bool operator<(const BinTableString &other) const;

  private:
    //In bytes
    uint32_t size = 0;
    char *data = nullptr;
    bool delete_data = false;
    uint32_t buffer_size = 0;


    void read_data_array(InputStream &stream);
};



NAMESPACE_END(NAMESPACE_BINTABLE)
