#ifndef MLPACK_PROJECT_PREDICT_REQUEST_DESERIALIZER_H
#define MLPACK_PROJECT_PREDICT_REQUEST_DESERIALIZER_H

#include <vector>
#include "../crow_all.h"
#include <mlpack.hpp>

using namespace mlpack;

class PredictRequestDeserializer {
private:
    data::DatasetInfo info;
    std::vector<std::string> dimensionToDataField;
public:
    PredictRequestDeserializer(
      data::DatasetInfo const &infoPtr,
      std::vector<std::string> const &dimensionToDataFieldPtr);

    void convertRequestBodyToInput(crow::json::rvalue &body, arma::colvec &input);
   
};
#endif // MLPACK_PROJECT_PREDICT_REQUEST_DESERIALIZER_H
