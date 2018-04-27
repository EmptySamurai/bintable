#pragma once
#include <cstdint>
#include <string>
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

const tabledatatype BINTABLE_OBJECT = 13;


const uint8_t DATATYPE_ELEMENT_SIZE[] = {1,1,1,2,2,4,4,8,8,4,8,1,4, 1};
const std::string DATATYPE_NAME[] = {"bool", "int8", "uint8", "int16", "uint16" ,
 "int32", "uint32", "int64", "uint64", "float32", "float64", "UTF-8", "UTF-32", "object"};


bool is_basic_bintable_datatype(tabledatatype type);

bool is_valid_datatype(tabledatatype type);

void validate_datatype(tabledatatype type);


NAMESPACE_END(NAMESPACE_BINTABLE)

