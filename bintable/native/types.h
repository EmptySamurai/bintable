#pragma once
#include <cstdint>
#include "common.h"


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

typedef uint8_t tabledatatype;


const tabledatatype BINTABLE_BOOL = 0;

const tabledatatype BINTABLE_INT8 = 1;
const tabledatatype BINTABLE_UINT8 = 2;

const tabledatatype BINTABLE_INT16 = 3;
const tabledatatype BINTABLE_UINT16 = 4;

const tabledatatype BINTABLE_INT32 = 5;
const tabledatatype BINTABLE_UINT32 = 6;

const tabledatatype BINTABLE_INT64 = 7;
const tabledatatype BINTABLE_UINT64 = 8;

const tabledatatype BINTABLE_FLOAT32 = 9;
const tabledatatype BINTABLE_FLOAT64 = 10;

const tabledatatype BINTABLE_UTF8 = 11;
const tabledatatype BINTABLE_UTF32 = 12;


const uint8_t BASIC_DATATYPES_SIZE[11] = {1,1,1,2,2,4,4,8,8,4,8};

bool is_basic_bintable_datatype(tabledatatype type);

NAMESPACE_END(NAMESPACE_BINTABLE)

