#pragma once
#include <cstdint>
#include "common.h"
#include "ioutils.h"


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)


class BinTableString { 
    public:
        //In bytes
        uint32_t size;
        char* data;
        bool delete_data;


        BinTableString();
        BinTableString(BufferedInputStream& stream);
        void write(BufferedOutputStream& stream);   
        static void read_to_buffer(BufferedInputStream& stream, char* buffer, uint32_t& size);
        BinTableString* copy();  
        ~BinTableString();

        bool operator==(const BinTableString& other);
        bool operator!=(const BinTableString& other);
};

NAMESPACE_END(NAMESPACE_BINTABLE)
