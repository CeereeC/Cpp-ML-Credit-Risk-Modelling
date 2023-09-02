#ifndef MLPACK_PROJECT_NN_H
#define MLPACK_PROJECT_NN_H

#include <mlpack/core/util/arma_traits.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/methods/ann/ffn.hpp>

void generateBaseFNN(

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

  // data::Save("data/scalar.bin", "scalar", scaleX, true);

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

  // data::Load("models/nn.bin", "nn", model);
  // data::Save("models/nn.bin", "nn", model, true);  
  std::cout << "FNN generated!" <<'\n';
}

#endif //MLPACK_PROJECT_NN_H
