#pragma once
#include <cstdint>
#include <vector>
#include "common.h"
#include "tablestring.h"
#include "types.h"

NAMESPACE_BEGIN(NAMESPACE_BINTABLE)

const uint64_t CURRENT_VERSION = 1;

class BinTableColumnDefinition { 
    public:
        tabledatatype type = UINT32_MAX;
        BinTableString* name = nullptr; 
        //In bytes. Optional 
        uint32_t maxlen = 0;
        
        BinTableColumnDefinition();

    explicit BinTableColumnDefinition(BufferedInputStream& stream);

    BinTableColumnDefinition(const BinTableColumnDefinition& other);

    void write(BufferedOutputStream& stream) const;

    ~BinTableColumnDefinition();

    private:
        bool has_maxlen() const;
    
};

class BinTableHeader {
    public:
        uint32_t version = CURRENT_VERSION;
        uint64_t n_rows = 0;
        uint32_t n_columns = 0;
        std::vector<BinTableColumnDefinition> columns;

        BinTableHeader();
        BinTableHeader(const BinTableHeader& other);

    explicit BinTableHeader(BufferedInputStream& stream);
        void write(BufferedOutputStream& stream) const;
 };

NAMESPACE_END(NAMESPACE_BINTABLE)