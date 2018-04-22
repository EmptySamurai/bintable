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
        tabledatatype type;
        BinTableString* name; 
        //In bytes. Optional 
        uint32_t maxlen;
        
        BinTableColumnDefinition();

    explicit BinTableColumnDefinition(BufferedInputStream& stream);

    virtual void write(BufferedOutputStream& stream);

    virtual ~BinTableColumnDefinition();

    private:
        bool has_maxlen();
    
};

class BinTableHeader {
    public:
        uint32_t version;
        uint64_t n_rows;
        uint32_t n_columns;
        std::vector<BinTableColumnDefinition*>* columns;

        BinTableHeader();

    explicit BinTableHeader(BufferedInputStream& stream);
        void write(BufferedOutputStream& stream);
        ~BinTableHeader();
 };

NAMESPACE_END(NAMESPACE_BINTABLE)