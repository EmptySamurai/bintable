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
    try{
        stream.read(data, size);
    } catch (std::exception ex) {
        delete[] data;
        throw ex;
    }
    delete_data = true;
};

BinTableString::BinTableString(const BinTableString& other) {
    size = other.size;
    data = new char[size];
    std::copy(other.data, other.data+size, data);
    delete_data = true;
};

BinTableString::BinTableString() = default;

void BinTableString::write(BufferedOutputStream& stream) {
    stream.write_primitive(size);
    stream.write(data, size);
};

BinTableString::~BinTableString() {
    if (delete_data) {
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

    for (uint32_t i=0; i<size; i++) {
        if (data[i]!=other.data[i]) {
            return false;
        }
    }

    return true;
    
}

bool BinTableString::operator!=(const BinTableString& other) {
    return !((*this)==other);
}

bool BinTableString::operator<(const BinTableString& other) const {
    if (size != other.size) {
        return size<other.size;
    }

    for (uint32_t i=0; i<size; i++) {
        if (data[i]!=other.data[i]) {
            return data[i]<other.data[i];
        }
    }

    return false;
}