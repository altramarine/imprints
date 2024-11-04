#ifndef COLUMN_IMPRINTS_H__
#define COLUMN_IMPRINTS_H__

#include "main.h"
#include "imprints.h"
#include "queries.h"
#include "zonemaps.h"
#include <vector>
#include <string>
#include <stdexcept>

using namespace ColumnImprints;

template <typename VALUE_TYPE>
class Imprints {
public:
  Imprints(int blocksize = 64, int maxbins = 64, std::string type_name = std::string("unsigned long")) : blocksize_(blocksize), maxbins_(maxbins) {
    column_ = (Column *) malloc(sizeof(Column));
    for(int i = 0; i < type_name.length(); i ++) column_->type_name[i] = type_name[i];
    column_->type_name[type_name.length()] = '\0';
    // strcpy(column_->type_name, type_name.c_str());// column_->type_name = type_name;
    // std::cout << boost::typeindex::type_id<VALUE_TYPE>().pretty_name() << std::endl;
    // printf("typename is: %s\n", column_->type_name);
    if (strcmp(column_->type_name, "tinyint") == 0 || strcmp(column_->type_name, "boolean") == 0) {
        column_->coltype  = TYPE_bte;
        column_->min.bval = 127;
        column_->max.bval = -127;
    } else if (strcmp(column_->type_name, "char") == 0 || strcmp(column_->type_name,"smallint")== 0 || strcmp(column_->type_name, "short")== 0) {
        column_->coltype  = TYPE_sht;
        column_->min.sval = 32767;
        column_->max.sval = -32767;
    } else if (strcmp(column_->type_name, "decimal") == 0 || strcmp(column_->type_name, "int") == 0 || strcmp(column_->type_name, "date") == 0) {
        column_->coltype  = TYPE_int;
        column_->min.ival = INT_MAX;
        column_->max.ival = INT_MIN;
    } else if (strcmp(column_->type_name, "long") == 0 || strcmp(column_->type_name, "long int") == 0) {
        column_->coltype  = TYPE_lng;
        column_->min.lval = LONG_MAX;
        column_->max.lval = LONG_MIN;
    } else if (strcmp(column_->type_name, "float") == 0 || strcmp(column_->type_name, "real") == 0) {
        column_->coltype= TYPE_flt;
        column_->min.fval = FLT_MAX;
        column_->max.fval = FLT_MIN;
    } else if (strcmp(column_->type_name, "double") == 0 ) {
        column_->coltype  = TYPE_dbl;
        column_->min.dval = DBL_MAX;
        column_->max.dval = -DBL_MAX;
    } else if (strcmp(column_->type_name, "oid") == 0 || strcmp(column_->type_name, "unsigned long") == 0) {
        column_->coltype  = TYPE_oid;
        column_->min.ulval = ULONG_MAX;
        column_->max.ulval = 0;
    } else {
        printf("error: type [%s] not supported\n", column_->type_name);
        std::runtime_error("[column imprints]: type not supported");
    }
    // std::cout << "column initiated" << std::endl;
  }

  void bulkload(
    std::vector<VALUE_TYPE> &values) {
    // expects the pairs to be pre-sorted before performing bulk load
    // this->_index.bulk_load(values.begin(), values.end());
    // binning()
    column_->col = (char *)new VALUE_TYPE[values.size()];
    for(size_t i = 0; i < values.size(); i++) {
        ((VALUE_TYPE *)column_->col)[i] = values[i];
    }
    const int stride[14]= { 0,0,0,1,2,0,4,8,0,0,4,8,8,0};
    int vpp = PAGESIZE/stride[column_->coltype];
    if (vpp == 0) {
        printf("rows per pages is 0\n");
        std::runtime_error("rows per pages is 0");
        // return -1;
    }
    int pages = column_->colcount/vpp + 1;
    if (pages > MAX_IMPS) {
        printf("there are too many pages %ld\n", pages);
        std::runtime_error("column imprints: too many pages");
        // return -1;
    }
    column_->typesize = stride[column_->coltype];
    column_->colcount = values.size();
    index_ = create_imprints(column_, blocksize_, maxbins_, 1);
  }

  unsigned int * range_scan(VALUE_TYPE low, VALUE_TYPE high) {
        ValRecord low_, high_;
        switch (column_->coltype) {
            case TYPE_bte:
                low_.bval = low;
                high_.bval = high;
                break;
            case TYPE_sht:
                low_.sval = low;
                high_.sval = high;
                break;
            case TYPE_int:
                low_.ival = low;
                high_.ival = high;
                // setqueryrange(ival);
                break;
            case TYPE_lng:
                low_.lval = low;
                high_.lval = high;
                break;
            case TYPE_oid:
                low_.ulval = low;
                high_.ulval = high;
                // setqueryrange(ulval);
                break;
            case TYPE_flt:
                low_.fval = low;
                high_.fval = high;
                // setqueryrange(fval);
                break;
            case TYPE_dbl:
                low_.dval = low;
                high_.dval = high;
                // setqueryrange(dval);
        }
        uint32_t *result_data = new uint32_t[(column_->colcount + 31) / 32];
        memset(result_data, 0, sizeof(uint32_t) * ((column_->colcount + 31) / 32));
        imprints_simd_scan(column_, index_, low_, high_, nullptr, result_data);
        return result_data;
    }

  ~Imprints() {
    if(index_->dct != nullptr)
      delete index_->dct;
    if(index_->bounds != nullptr)
      delete index_->bounds;
    if(index_->imprints != nullptr)
      delete index_->imprints;
    delete index_;
    delete column_;
  }
private:
  Column *column_;
  Imprints_index *index_;
  int blocksize_, maxbins_;
};

#endif