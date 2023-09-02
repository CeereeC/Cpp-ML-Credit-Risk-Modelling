#include "ModelEvaluator.h"

// Utility functions for evaluation metrics.
double ModelEvaluator::ComputeAccuracy(const arma::Row<double>& yPreds, const arma::Row<double>& yTrue) {
  const double correct = arma::accu(yPreds == yTrue);
  return (double)correct / yTrue.n_elem;
}

double ModelEvaluator::ComputePrecision(const double truePos, const double falsePos) {
  return (double)truePos / (double)(truePos + falsePos);
}

double ModelEvaluator::ComputeRecall(const double truePos, const double falseNeg) {
  return (double)truePos / (double)(truePos + falseNeg);
}

double ModelEvaluator::ComputeF1Score(const double truePos, const double falsePos, const double falseNeg) {
  double prec = ComputePrecision(truePos, falsePos);
  double rec = ComputeRecall(truePos, falseNeg);
  return 2 * (prec * rec) / (prec + rec);
}

std::string ModelEvaluator::ClassificationReport(const arma::Row<double>& yPreds, const arma::Row<double>& yTrue) {
  arma::Row<double> uniqs = arma::unique(yTrue);

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

