#include "streams/constantstream.h"
#include <algorithm>

using namespace NAMESPACE_BINTABLE;

ConstantInputStream::ConstantInputStream(char value) : value(value) {

}

void ConstantInputStream::read(char *data, const std::streamsize size) {
    std::fill_n(data, size, value);
}

void ConstantInputStream::skip(const std::streamsize size) {

}


