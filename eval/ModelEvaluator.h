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

    template<typename PredType, typename TrueType>
    static std::string ClassificationReport(const PredType& yPreds, const TrueType& yTrue) {
      TrueType uniqs = arma::unique(yTrue);

      std::ostringstream out;

      out << std::setw(14) << "precision" << std::setw(15) << "recall"
        << std::setw(15) << "f1-score" << std::setw(15) << "support"
        << '\n' << '\n';

      for(auto val: uniqs) {
        double truePos = arma::accu(yTrue == val && yPreds == val && yPreds == yTrue);
        double falsePos = arma::accu(yPreds == val && yPreds != yTrue);
        double trueNeg = arma::accu(yTrue != val && yPreds != val && yPreds == yTrue);
        double falseNeg = arma::accu(yPreds != val && yPreds != yTrue);

        out<< val
          << std::setw(12) << std::setprecision(2) << ComputePrecision(truePos, falsePos)
          << std::setw(16) << std::setprecision(2) << ComputeRecall(truePos, falseNeg)
          << std::setw(14) << std::setprecision(2) << ComputeF1Score(truePos, falsePos, falseNeg)
          << std::setw(16) << truePos
          << '\n';
      }

      return out.str();
    }


    template<typename MLAlgorithm, typename DataType, typename ResponsesType>
      static std::string Eval(
          MLAlgorithm& model,
          const DataType& data,
          const ResponsesType& labels) {

        arma::rowvec predictions;
        model.Predict(data, predictions);
        arma::rowvec predY = round(predictions);  // Set threshold

        return ClassificationReport(predY, labels);
      }    

};


#endif //MLPACK_PROJECT_MODEL_EVAL_H
