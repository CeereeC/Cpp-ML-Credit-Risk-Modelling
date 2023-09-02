#include "ModelGenerator.h"

void ModelGenerator::generateModels() {
    arma::mat dataset;  
    data::DatasetInfo info;
    data::Load("data/cleaned_credit_data.csv", dataset, info); // Remember that Load(...) transposes the matrix
    data::Save("data/dataset_info.bin", "dataset_info", info, true);

    // ============ Preprocess Data ============= //

    arma::mat trainData, testData;
    data::Split(dataset, trainData, testData, 0.1);

    arma::mat trainX = trainData.submat(0, 0, trainData.n_rows - 2, trainData.n_cols - 1);
    arma::mat testX = testData.submat(0, 0, testData.n_rows - 2, testData.n_cols - 1);

    arma::mat trainY = trainData.row(trainData.n_rows - 1);
    arma::mat testY = testData.row(testData.n_rows - 1);

    // ============ Linear Regression ============= //
    generateBaseLinReg(trainX, trainY, testX, testY);
    runTunedLinReg(trainX, trainY, testX, testY);

    // ============ Neural Network ============= //

    generateBaseFNN(trainX, trainY, testX, testY, trainData.n_cols);
}

void ModelGenerator::generateBaseLinReg(
    const arma::mat &trainX, 
    const arma::mat &trainY,
    const arma::mat &testX, 
    const arma::mat &testY) {

  LinearRegression lr(trainX, trainY);
  data::Save("models/lr.bin", "lr", lr, true);
  std::cout << "Linear Regression Model generated!";
}

void ModelGenerator::runTunedLinReg(
    const arma::mat &trainX, 
    const arma::mat &trainY,
    const arma::mat &testX, 
    const arma::mat &testY) {
  
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

void ModelGenerator::generateBaseFNN(
    const arma::mat &trainX, 
    const arma::mat &trainY,
    const arma::mat &testX, 
    const arma::mat &testY,
    size_t num_data) {

  // Scale all data into the range (0, 1) for increased numerical stability.
  data::MinMaxScaler scaleX;
  arma::mat scTrainX, scTestX;
  scaleX.Fit(trainX);
  scaleX.Transform(trainX, scTrainX);
  scaleX.Transform(testX, scTestX);

  // Save Scalar to reuse when transforming input
  data::Save("data/scalar.bin", "scalar", scaleX, true);

  const int EPOCHS = 30;
  constexpr double STEP_SIZE = 5e-2;
  constexpr int BATCH_SIZE = 32;
  constexpr double STOP_TOLERANCE = 1e-8;

  // ========== Feed Forward Neural Network ========== /

  FFN<MeanSquaredError, RandomInitialization> model;
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
      num_data * EPOCHS, // Max number of iterations.
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

  data::Save("models/nn.bin", "nn", model, true);  
  std::cout << "FNN generated!" <<'\n';
}
