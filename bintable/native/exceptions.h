#pragma once
#include "common.h"
#include <stdexcept>

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)


class StreamFinishedException : public std::exception {
    using std::exception::exception;
};

class WrongPythonObjectException : public std::exception {
    using std::exception::exception;
};

class UnknownDatatypeException : public std::exception {
    using std::exception::exception;
};

NAMESPACE_END(NAMESPACE_BINTABLE)

