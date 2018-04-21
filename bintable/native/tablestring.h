#pragma once
#include <cstdint>
#include <istream>
#include "common.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)


class BinTableString { 
    public:
        //In bytes
        uint32_t size;
        char* data;
        bool deleteData;


        BinTableString();
        BinTableString(std::istream& stream);
        void write(std::ostream& stream);   
        static void read_to_buffer(std::istream& stream, char* buffer, uint32_t& size);
        BinTableString* copy();  
        ~BinTableString();

        bool operator==(const BinTableString& other);
        bool operator!=(const BinTableString& other);
};

NAMESPACE_END(NAMESPACE_BINTABLE)
