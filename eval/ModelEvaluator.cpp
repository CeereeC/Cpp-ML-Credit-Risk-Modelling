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

