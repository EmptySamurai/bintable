#include "tablestring.h"
#include "ioutils.h"

using namespace NAMESPACE_BINTABLE;

void BinTableString::read_to_buffer(BufferedInputStream& stream, char* buffer, uint32_t& size) {
    stream.read_primitive(size);
    stream.read(buffer, size);
}


BinTableString::BinTableString(BufferedInputStream& stream) {
    stream.read_primitive(size);
    data = new char[size];
    stream.read(data, size);
    deleteData = true;
};

BinTableString::BinTableString() {
    deleteData = false;
};

void BinTableString::write(BufferedOutputStream& stream) {
    stream.write_primitive(size);
    stream.write(data, size);
};

BinTableString* BinTableString::copy() {
    auto result = new BinTableString();
    char* new_data = new char[size];
    std::copy(data, data+size, new_data);

    result->size = size;
    result->data = new_data;
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
