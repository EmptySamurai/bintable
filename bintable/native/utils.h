#pragma once
#include <istream>

//MAYBE ADD CHECKS

template <class T>
void read_from_stream(std::istream& stream, T& val) {
    stream.read(reinterpret_cast<char *>(&val), sizeof(val));
}

template <class T>
void write_to_stream(std::ostream& stream, T& val) {
    stream.write(reinterpret_cast<char *>(&val), sizeof(val));
}
