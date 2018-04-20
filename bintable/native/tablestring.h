#pragma once
#include <cstdint>
#include <istream>
#include "common.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)


class BinTableString { 
    public:
        uint32_t size;
        char* data;

        BinTableString();
        BinTableString(std::istream& stream);
        void write(std::ostream& stream);     
        ~BinTableString();

        bool operator==(const BinTableString& other);
        bool operator!=(const BinTableString& other);

    private:
        bool deleteData;
};

NAMESPACE_END(NAMESPACE_BINTABLE)
