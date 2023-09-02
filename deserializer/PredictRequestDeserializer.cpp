#include "PredictRequestDeserializer.h"

PredictRequestDeserializer::PredictRequestDeserializer(
      data::DatasetInfo const &infoPtr,
      std::vector<std::string> const &dimensionToDataFieldPtr): info(infoPtr),dimensionToDataField(dimensionToDataFieldPtr){}

void PredictRequestDeserializer::convertRequestBodyToInput(crow::json::rvalue &body, arma::colvec &input) {
  for (size_t i = 0; i < input.size(); ++i) {
    auto field = body[dimensionToDataField[i]];
    std::string ss;
    if (field.t() == crow::json::type::String) {
      ss = field.s();   // If you don't do this, it'll return "abc" with the quotes
    } else {
      std::ostringstream os;
      os << field;
      ss = os.str();
    }
    input(i) = info.MapString<double>(ss, i);
  }
}
