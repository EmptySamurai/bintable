#pragma once
#include "common.h"
#include "streams/streams.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class MemoryInputStream : public InputStream {
    public:
        MemoryInputStream(char* input_data, std::streamsize input_data_size);
        void read(char *data, const std::streamsize size) override;
        void skip(const std::streamsize size) override;
    
    private:
        char* input_data;
        const std::streamsize input_data_size;
        std::streamsize current_position;
};

class MemoryOutputStream : public OutputStream {
    public:
        MemoryOutputStream(char* output_data, std::streamsize output_data_max_size);
        void write(const char *data, const std::streamsize size) override;
        void write(InputStream& stream, const std::streamsize size) override;
    
    private:
        char* output_data;
        const std::streamsize output_data_max_size;
        std::streamsize current_position;
};

NAMESPACE_END(NAMESPACE_BINTABLE)
