#ifndef MLPACK_PROJECT_LINREG_H
#define MLPACK_PROJECT_LINREG_H

#include <iostream>
#include <mlpack/methods/linear_regression.hpp>
#include <mlpack/core/util/arma_traits.hpp>
#include <mlpack/core/hpt/hpt.hpp>
#include "ModelEvaluator.h"

using namespace mlpack;

void runBaseLinReg(
    const arma::mat &trainX, 
    const arma::mat &trainY,
    const arma::mat &testX, 
    const arma::mat &testY) {

  LinearRegression lr(trainX, trainY);
  ModelEvaluator::Evaluate(lr, testX, testY);

  // data::Load("models/lr.bin", "lr", lr);
  // data::Save("models/lr.bin", "lr", lr, true);
}


void runTunedLinReg(
    const arma::mat &trainX, 
    const arma::mat &trainY,
    const arma::mat &testX, 
    const arma::mat &testY) {
  // ============ Tuning ============== //
  // Using 80% of data for training and remaining 20% for assessing MSE.
  double validationSize = 0.2;
  HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(validationSize,
      trainX, trainY);

  // Finding a good value for lambda from the discrete set of values 0.0, 0.001,
  // 0.01, 0.1, and 1.0.
  arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
  double bestLambda;
  std::tie(bestLambda) = hpt.Optimize(lambdas);
  std::cout << "Best Lambda: " << bestLambda << '\n';

}


#endif //MLPACK_PROJECT_LINREG_H
