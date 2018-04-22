#pragma once
#include <iostream>

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

#if !defined(PRINT)
#  define PRINT(val) std::cout<<val<<std::endl;
#endif