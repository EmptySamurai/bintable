#include "common.h"
#include "types.h"
#include "exceptions.h"


bool NAMESPACE_BINTABLE::is_basic_bintable_datatype(tabledatatype type) {
    return type<=10;
}

bool NAMESPACE_BINTABLE::is_valid_datatype(tabledatatype type){
    return type<=13;
}

void NAMESPACE_BINTABLE::validate_datatype(tabledatatype type) {
    if (!is_valid_datatype(type)) {
        throw UnknownDatatypeException("Unknow bintable datatype " + std::to_string((int)type));
    }
}
