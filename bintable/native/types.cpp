#include "common.h"
#include "types.h"
#include "exceptions.h"


bool NAMESPACE_BINTABLE::is_basic_bintable_datatype(tabledatatype type) {
    return type<=10;
}

bool NAMESPACE_BINTABLE::is_valid_datatype(tabledatatype type){
    return type<=13;
}
