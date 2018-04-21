#pragma once
#include "common.h"
#include <istream>
#include <stdexcept>


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

extern const int _BUFFER_MAX_SIZE;
extern char _BUFFER [];
extern std::streamsize _BUFFER_SIZE;
extern std::streamsize _CURRENT_BUFFER_POSITION;



inline void flush_buffer(std::ostream &stream) {
    stream.write(_BUFFER, _BUFFER_SIZE);
    _BUFFER_SIZE = 0;
};

inline void write_to_stream_buffered(std::ostream &stream, char* data, std::streamsize size) {
    if (_BUFFER_SIZE + size>_BUFFER_MAX_SIZE) {
        flush_buffer(stream);
        stream.write(data, size);
    } else {
        std::copy(data, data+size, _BUFFER+_BUFFER_SIZE);
        _BUFFER_SIZE+=size;
    }
};

template <class T>
inline void write_primitive_to_stream(std::ostream& stream, T& val) {
    //TODO: add support for endianess
    write_to_stream_buffered(stream, reinterpret_cast<char *>(&val), sizeof(val));
};


inline void clear_buffer() {
    _BUFFER_SIZE = 0;
    _CURRENT_BUFFER_POSITION = 0;
};

inline void read_from_stream_buffered(std::istream &stream, char* data, std::streamsize size) {
    std::streamsize bytesLeftInBuffer = _BUFFER_SIZE - _CURRENT_BUFFER_POSITION;
    if (bytesLeftInBuffer<size) {
        std::copy(_BUFFER+_CURRENT_BUFFER_POSITION, _BUFFER+_BUFFER_SIZE, data);
        std::streamsize bytesLeftNotWritten = size-bytesLeftInBuffer;
        stream.read(data+bytesLeftInBuffer, bytesLeftNotWritten);

        if (bytesLeftNotWritten != stream.gcount()) {
            throw std::length_error("Not enough bytes to read");
        }


        stream.read(_BUFFER, _BUFFER_MAX_SIZE);
        _CURRENT_BUFFER_POSITION=0;
        _BUFFER_SIZE = stream.gcount();
    } else {
        std::copy(_BUFFER+_CURRENT_BUFFER_POSITION, _BUFFER+_CURRENT_BUFFER_POSITION+size, data);
        _CURRENT_BUFFER_POSITION+=size;
    }
};



template <class T>
void read_primitive_from_stream(std::istream& stream, T& val) {
    read_from_stream_buffered(stream, reinterpret_cast<char *>(&val), sizeof(val));
};

NAMESPACE_END(NAMESPACE_BINTABLE)


