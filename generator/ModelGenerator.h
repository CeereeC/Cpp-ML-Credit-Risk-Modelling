#ifndef MLPACK_PROJECT_MODEL_GEN_H
#define MLPACK_PROJECT_MODEL_GEN_H

#include <mlpack.hpp>
#include <iostream>


using namespace mlpack;

class ModelGenerator {
public:
  static void generateModels();

  static void generateBaseLinReg(
    const arma::mat &trainX, 
    const arma::mat &trainY);

  static void runTunedLinReg(
    const arma::mat &trainX, 
    const arma::mat &trainY);

  static void generateBaseFNN(
    const arma::mat &trainX, 
    const arma::mat &trainY,
    size_t num_data);

  static void generateBaseDT(
    const arma::mat &trainX, 
    const arma::mat &trainY);
};

#endif //MLPACK_PROJECT_MODEL_GEN_H
