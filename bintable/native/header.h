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

    virtual void write(BufferedOutputStream& stream);

    virtual ~BinTableColumnDefinition();

    private:
        bool has_maxlen();
    
};

class BinTableHeader {
    public:
        uint32_t version = CURRENT_VERSION;
        uint64_t n_rows;
        uint32_t n_columns;
        std::vector<BinTableColumnDefinition*>* columns = nullptr;

        BinTableHeader();
        BinTableHeader(const BinTableHeader& other);

    explicit BinTableHeader(BufferedInputStream& stream);
        void write(BufferedOutputStream& stream);
        void delete_columns();
        ~BinTableHeader();
 };

NAMESPACE_END(NAMESPACE_BINTABLE)