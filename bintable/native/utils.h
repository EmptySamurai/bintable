#pragma once
#include "common.h"
#include <istream>


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

extern const int _BUFFER_MAX_SIZE;
extern char _BUFFER [];
extern std::streamsize _BUFFER_SIZE;


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
    }
};

template <class T>
inline void write_primitive_to_stream(std::ostream& stream, T& val) {
    write_to_stream_buffered(stream, reinterpret_cast<char *>(&val), sizeof(val));
};

template <class T>
void read_primitive_from_stream(std::istream& stream, T& val) {
    stream.read(reinterpret_cast<char *>(&val), sizeof(val));
};

NAMESPACE_END(NAMESPACE_BINTABLE)


