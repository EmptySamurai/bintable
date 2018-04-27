#pragma once

#include <vector>
#include <string>

#include "header.h"
#include "types.h"
#include "common.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

struct BinTableColumnData
{
    //In symbols
    uint64_t size;
    tabledatatype type;
    char *data = nullptr;
    BinTableString *name = nullptr;
    //In bytes
    uint32_t maxlen;
};

void write_table(std::vector<BinTableColumnData *> &data, const std::string &path, bool append);

void read_table(const std::string &path, std::vector<BinTableColumnData *> &out);

BinTableHeader read_header(const std::string &path);


NAMESPACE_END(NAMESPACE_BINTABLE)
