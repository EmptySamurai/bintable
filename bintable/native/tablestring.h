#pragma once
#include <cstdint>
#include <string>
#include "common.h"
#include "ioutils.h"

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
    explicit BinTableString(BufferedInputStream &stream);
    void write(BufferedOutputStream &stream);
    static void read_to_buffer(BufferedInputStream &stream, char *buffer, uint32_t &size);
    std::string to_string() const;
    ~BinTableString();

    bool operator==(const BinTableString &other);
    bool operator!=(const BinTableString &other);
    bool operator<(const BinTableString &other) const;
};



NAMESPACE_END(NAMESPACE_BINTABLE)
