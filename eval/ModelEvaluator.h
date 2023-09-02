#ifndef MLPACK_PROJECT_MODEL_EVAL_H
#define MLPACK_PROJECT_MODEL_EVAL_H

#include <iostream>
#include <mlpack.hpp>
#include <type_traits>
#include <string>

class ModelEvaluator {
  public:

    // Utility functions for evaluation metrics.
    static double ComputeAccuracy(const arma::Row<double>& yPreds, const arma::Row<double>& yTrue);

    static double ComputePrecision(const double truePos, const double falsePos);

    static double ComputeRecall(const double truePos, const double falseNeg);

    static double ComputeF1Score(const double truePos, const double falsePos, const double falseNeg);

    static std::string ClassificationReport(const arma::Row<double>& yPreds, const arma::Row<double>& yTrue);

    template<typename MLAlgorithm, typename DataType, typename ResponsesType>
    static std::string Eval(
          const MLAlgorithm& model,
          const DataType& data,
          const ResponsesType& labels) {

        arma::rowvec predictions;
        model.Predict(data, predictions);
        arma::rowvec predY = round(predictions);  // Set threshold

        return ClassificationReport(predY, labels);
      }    
};


#endif //MLPACK_PROJECT_MODEL_EVAL_H
