
#include "common.h"
#include "types.h"

bool NAMESPACE_BINTABLE::is_basic_bintable_datatype(tabledatatype type) {
    return type<=10;
}