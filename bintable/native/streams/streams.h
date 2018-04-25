#pragma once
#include "common.h"
#include "exceptions.h"
#include <algorithm>
#include <iostream>

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

template<class T> 
inline void reverse_bytes(T& val) {
    std::reverse(reinterpret_cast<char *>(&val), reinterpret_cast<char *>(&val) + sizeof(val));
}

class InputStream
{
    public:
        virtual void read(char *data, const std::streamsize size)=0;
        virtual void skip(const std::streamsize size)=0;

        template <class T>
        void read_primitive(T &val) {
            read(reinterpret_cast<char *>(&val), sizeof(val));
        }

        template <class T>
        void read_primitive_endian_aware(T &val) {
            read_primitive(val);
            #if !(USE_LITTLE_ENDIAN == SYSTEM_IS_LITTLE_ENDIAN)
                reverse_bytes(val);
            #endif
        }

        virtual ~InputStream() = default;

    protected:
        void throw_not_enough_bytes() {
            throw StreamFinishedException("Not enough bytes to read");
        }
};

class OutputStream
{
    public:
        virtual void write(const char *data, const std::streamsize size) = 0;
        virtual void write(InputStream& stream, const std::streamsize size) = 0;

        template <class T>
        void write_primitive(const T &val) {
            write(reinterpret_cast<const char *>(&val), sizeof(val));
        }

        template <class T>
        void write_primitive_endian_aware(const T &val) {
            #if USE_LITTLE_ENDIAN == SYSTEM_IS_LITTLE_ENDIAN
                write_primitive(val);
            #else
                T copy = val;
                reverse_bytes(copy);
                write_primitive(copy);
            #endif
        }

        virtual ~OutputStream() = default;

};

NAMESPACE_END(NAMESPACE_END)

