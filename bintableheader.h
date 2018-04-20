#pragma once
#include <cstdint>
#include <istream>
#include <vector>
#include "bintablestring.h"
#include "bintabletypes.h"

const uint64_t CURRENT_VERSION = 1;

class BinTableColumnDefinition { 
    public:
        tabledatatype type;
        BinTableString* name; 
        
        BinTableColumnDefinition();

    explicit BinTableColumnDefinition(std::istream& stream);

    virtual void write(std::ostream& stream);

    virtual ~BinTableColumnDefinition();
    
};

class BinTableStringColumnDefinition : BinTableColumnDefinition {
public:
    uint32_t maxlen;

    BinTableStringColumnDefinition();

    explicit BinTableStringColumnDefinition(std::istream& stream);
    void write(std::ostream& stream) override;
    ~BinTableStringColumnDefinition();

};

class BinTableHeader {
    public:
        uint32_t version;
        uint64_t n_rows;
        uint32_t n_columns;
        std::vector<BinTableColumnDefinition*>* columns;

        BinTableHeader();

    explicit BinTableHeader(std::istream& stream);
        void write(std::ostream& stream);
        ~BinTableHeader();
 };

