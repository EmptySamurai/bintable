#pragma once
#include "common.h"
#include "streams/streams.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class ConstantInputStream : public InputStream {
    public:
        ConstantInputStream(char value);
        void read(char *data, const std::streamsize size) override;
        void skip(const std::streamsize size) override;
    
    private:
        const char value;
};

NAMESPACE_END(NAMESPACE_BINTABLE)
