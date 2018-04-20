#pragma once

#include <vector>
#include <string>

#include "header.h"
#include "types.h"
#include "common.h"


NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

struct BinTableColumnData {
    uint64_t size;
    tabledatatype type;
    char *data;
    BinTableString *name;
    uint32_t maxlen;
};

void write_table(std::vector<BinTableColumnData *> &data, const std::string &path, bool append);

NAMESPACE_END(NAMESPACE_BINTABLE)





