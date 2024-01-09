#include "ModelGenerator.h"


ModelGenerator::ModelGenerator(arma::mat &dataset) {

  // ============ Preprocess Data ============= //
  arma::mat trainData, testData;
  data::Split(dataset, trainData, testData, 0.1);

  trainX = trainData.submat(0, 0, trainData.n_rows - 2, trainData.n_cols - 1);
  trainY = trainData.row(trainData.n_rows - 1);
}

void ModelGenerator::generateBaseLinReg() {

  LinearRegression lr(trainX, trainY);
  data::Save("models/lr.bin", "lr", lr, true);
  std::cout << "Linear Regression Model generated!" << '\n';
}

void ModelGenerator::runTunedLinReg() {

  // Using 80% of data for training and remaining 20% for assessing MSE.
  double validationSize = 0.2;
  HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(validationSize,
      trainX, trainY);

  // Finding a good value for lambda from the discrete set of values 0.0, 0.001, 0.01, 0.1, and 1.0.
  arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
  double bestLambda;
  std::tie(bestLambda) = hpt.Optimize(lambdas);
  std::cout << "Best Lambda: " << bestLambda << '\n';

}

void ModelGenerator::generateBaseDT() {

  arma::Row<size_t> dataY = arma::conv_to<arma::Row<size_t>>::from(trainY);
  DecisionTree<> dt(trainX, dataY, 2);
  data::Save("models/dt.bin", "dt", dt, true);
  std::cout << "Decision Tree Model generated!" << '\n';
}

void ModelGenerator::generateBaseFNN(
    FFN<MeanSquaredError, RandomInitialization> &model) {

  // Scale all data into the range (0, 1) for increased numerical stability.
  data::MinMaxScaler scaleX;
  arma::mat scTrainX;
  scaleX.Fit(trainX);
  scaleX.Transform(trainX, scTrainX);

  // Save Scalar to reuse when transforming input
  data::Save("data/scalar.bin", "scalar", scaleX, true);

  const int EPOCHS = 1000;
  constexpr double STEP_SIZE = 5e-2;
  constexpr int BATCH_SIZE = 32;
  constexpr double STOP_TOLERANCE = 1e-8;

  // ========== Feed Forward Neural Network ========== /
  model.Reset();  
  model.Add<Linear>(32);
  model.Add<FlexibleReLU>();
  model.Add<Linear>(16);
  model.Add<Sigmoid>();
  model.Add<Linear>(1);

  // Optimizer
  ens::Adam optimizer(
      STEP_SIZE,  
      BATCH_SIZE, 
      0.9,        // Exponential decay rate for the first moment estimates.
      0.999,      // Exponential decay rate for the weighted infinity norm estimates.
      1e-8,       // Value used to initialise the mean squared gradient parameter.
      EPOCHS,     // Max number of iterations.
      1e-8,       // Tolerance.
      true);

  // Train the model.
  model.Train(scTrainX,
      trainY,
      optimizer,
      ens::PrintLoss(),
      ens::ProgressBar(40), // 40 is the width
                            // Stops the optimization process if the loss stops decreasing
                            // or no improvement has been made. This will terminate the
                            // optimization once we obtain a minima on training set.
      ens::EarlyStopAtMinLoss(20)); 

  std::cout << "FNN generated!" <<'\n';
}
