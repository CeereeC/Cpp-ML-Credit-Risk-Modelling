#ifndef MLPACK_PROJECT_MODEL_GEN_H
#define MLPACK_PROJECT_MODEL_GEN_H

#include <mlpack.hpp>
#include <iostream>


using namespace mlpack;

class ModelGenerator {
private:
  arma::mat trainX;
  arma::mat trainY;
public:
  
  ModelGenerator(arma::mat &dataset);

  void generateBaseLinReg();
  void generateBaseFNN(FFN<MeanSquaredError, RandomInitialization> &model);
  void generateBaseDT();
  void runTunedLinReg();
};

#endif //MLPACK_PROJECT_MODEL_GEN_H
