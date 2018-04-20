#pragma once
#include <cstdint>
#include <istream>

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
