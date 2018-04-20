#include "tablestring.h"
#include "utils.h"

using namespace NAMESPACE_BINTABLE;


BinTableString::BinTableString(std::istream& stream) {
    read_primitive_from_stream(stream, size);
    data = new char[size];
    stream.read(data, size);
    deleteData = true;
};

BinTableString::BinTableString() {
    deleteData = false;
};

void BinTableString::write(std::ostream& stream) {
    write_primitive_to_stream(stream, size);
    stream.write(data, size);
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
