#pragma once
#include "common.h"
#include <stdexcept>

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

class BinTableException : public std::exception {
public:
    explicit BinTableException(const char * m) : message{m} {}
    explicit BinTableException(const std::string& m) : message{m} {}
    virtual const char * what() const noexcept override {return message.c_str();}
private:
    std::string message = "";
};

class NotBinTableException : public BinTableException {
    using BinTableException::BinTableException;
};


class StreamFinishedException : public BinTableException {
    using BinTableException::BinTableException;
};

class WrongPythonObjectException : public BinTableException {
    using BinTableException::BinTableException;
};

class UnknownDatatypeException : public BinTableException {
    using BinTableException::BinTableException;
};

class AppendException : public BinTableException {
    using BinTableException::BinTableException;
};


NAMESPACE_END(NAMESPACE_BINTABLE)

