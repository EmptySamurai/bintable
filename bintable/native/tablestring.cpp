#include "tablestring.h"
#include "streams/streams.h"
#include <sstream>

using namespace NAMESPACE_BINTABLE;

void BinTableString::read_to_buffer(InputStream& stream, char* buffer, uint32_t& size) {
    stream.read_primitive_endian_aware(size);
    stream.read(buffer, size);
}

void BinTableString::read_to_stream(InputStream &input_stream, OutputStream &output_stream, uint32_t &size) {
    input_stream.read_primitive_endian_aware(size);
    output_stream.write(input_stream, size);
}

void BinTableString::read_data_array(InputStream& stream) {
    data = new char[size];
    try{
        stream.read(data, size);
    } catch (std::exception ex) {
        delete[] data;
        throw ex;
    }
    delete_data = true;
}

BinTableString::BinTableString(InputStream& stream) {
    stream.read_primitive_endian_aware(size);
    read_data_array(stream);
};

BinTableString::BinTableString(InputStream& stream, uint32_t size) : size(size) {
    read_data_array(stream);
};

BinTableString::BinTableString(const BinTableString& other) {
    size = other.size;
    data = new char[size];
    std::copy(other.data, other.data+size, data);
    delete_data = true;
};

BinTableString::BinTableString() = default;

void BinTableString::write(OutputStream& stream) {
    stream.write_primitive_endian_aware(size);
    stream.write(data, size);
};

void BinTableString::skip(InputStream &stream, uint32_t &size) {
    stream.read_primitive_endian_aware(size);
    stream.skip(size);
};

std::string BinTableString::to_string() const {
    std::stringstream stream;
    stream.write(data, size);
    return stream.str();
}

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