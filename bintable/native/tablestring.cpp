#include "tablestring.h"
#include "ioutils.h"

using namespace NAMESPACE_BINTABLE;


void BinTableString::read_to_buffer(std::istream& stream, char* buffer) {
    uint32_t size = 0;
    read_primitive_from_stream(stream, size);
    read_from_stream_buffered(stream, buffer, size);
}


BinTableString::BinTableString(std::istream& stream) {
    read_primitive_from_stream(stream, size);
    data = new char[size];
    read_from_stream_buffered(stream, data, size);
    deleteData = true;
};

BinTableString::BinTableString() {
    deleteData = false;
};

void BinTableString::write(std::ostream& stream) {
    write_primitive_to_stream(stream, size);
    write_to_stream_buffered(stream, data, size);
};

BinTableString* BinTableString::copy() {
    auto result = new BinTableString();
    char* newData = new char[size];
    std::copy(data, data+size, newData);

    result->size = size;
    result->data = newData;
    result->deleteData = true;
    return result;
};

BinTableString::~BinTableString() {
    if (deleteData) {
        delete[] data; 
    }   
}

bool BinTableString::operator==(const BinTableString& other) {
    if ((data==nullptr) || (other.data == nullptr)) {
        if ((data==nullptr) && (other.data == nullptr)) {
            return true;
        } else {
            return false;
        }
    }

    if (size != other.size) {
        return false;
    }

    for (auto i=0; i<size; i++) {
        if (data[i]!=other.data[i]) {
            return false;
        }
    }

    return true;
    
}

bool BinTableString::operator!=(const BinTableString& other) {
    return !((*this)==other);
}
