#include <iostream>
#include <mlpack.hpp>

using namespace mlpack;

// Utility functions for evaluation metrics.
double ComputeAccuracy(const arma::Row<double>& yPreds, const arma::Row<double>& yTrue)
{
  const double correct = arma::accu(yPreds == yTrue);
  return (double)correct / yTrue.n_elem;
}

double ComputePrecision(const double truePos, const double falsePos)
{
  return (double)truePos / (double)(truePos + falsePos);
}


double ComputeRecall(const double truePos, const double falseNeg)
{
  return (double)truePos / (double)(truePos + falseNeg);
}


double ComputeF1Score(const double truePos, const double falsePos, const double falseNeg)
{
  double prec = ComputePrecision(truePos, falsePos);
  double rec = ComputeRecall(truePos, falseNeg);
  return 2 * (prec * rec) / (prec + rec);
}


void ClassificationReport(const arma::Row<double>& yPreds, const arma::Row<double>& yTrue)
{
  arma::Row<double> uniqs = arma::unique(yTrue);
  std::cout << std::setw(29) << "precision" << std::setw(15) << "recall"
    << std::setw(15) << "f1-score" << std::setw(15) << "support"
    << std::endl << std::endl;

  for(auto val: uniqs)
  {
    double truePos = arma::accu(yTrue == val && yPreds == val && yPreds == yTrue);
    double falsePos = arma::accu(yPreds == val && yPreds != yTrue);
    double trueNeg = arma::accu(yTrue != val && yPreds != val && yPreds == yTrue);
    double falseNeg = arma::accu(yPreds != val && yPreds != yTrue);

    std::cout << std::setw(15) << val
      << std::setw(12) << std::setprecision(2) << ComputePrecision(truePos, falsePos)
      << std::setw(16) << std::setprecision(2) << ComputeRecall(truePos, falseNeg)
      << std::setw(14) << std::setprecision(2) << ComputeF1Score(truePos, falsePos, falseNeg)
      << std::setw(16) << truePos
      << std::endl;
  }
}


int main() {

  // Testing data is taken from the dataset in this ratio.
  constexpr double RATIO = 0.1; //10%

  arma::mat dataset;  
  data::DatasetInfo info;
  data::Load("data/cleaned_credit_data.csv", dataset, info); // Remember that Load(...) transposes the matrix


  // ============ Preprocess Data ============= //

  arma::mat trainData, testData;
  data::Split(dataset, trainData, testData, RATIO);

  arma::mat trainX = trainData.submat(0, 0, trainData.n_rows - 2, trainData.n_cols - 1);
  arma::mat testX = testData.submat(0, 0, testData.n_rows - 2, testData.n_cols - 1);

  arma::mat trainY = trainData.row(trainData.n_rows - 1);
  arma::mat testY = testData.row(testData.n_rows - 1);

  //  // ============ Linear Regression ============= //
  
  //  // Regress.
  //  LinearRegression lr(trainX, trainY);
  
  //  // This will store the predictions; one row for each point.
  //  arma::rowvec predictions;
  //  lr.Predict(testX, predictions);   // Predict.
  //  arma::rowvec predY = round(predictions);  // Set threshold
  
  //  // Test
  //  ClassificationReport(predY, testY);

  //  // ============ Tuning ============== //
  // // Using 80% of data for training and remaining 20% for assessing MSE.
  // double validationSize = 0.2;
  // HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(validationSize,
  //     trainX, trainY);

  // // Finding a good value for lambda from the discrete set of values 0.0, 0.001,
  // // 0.01, 0.1, and 1.0.
  // arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
  // double bestLambda;
  // std::tie(bestLambda) = hpt.Optimize(lambdas);
  // std::cout << bestLambda << '\n';

   // ============ Neural Network ============= //
  
   // Scale all data into the range (0, 1) for increased numerical stability.
   data::MinMaxScaler scaleX;
   // Scaler for predictions.
   data::MinMaxScaler scaleY;
  
   arma::mat scTrainX, scTestX, scTrainY, scTestY;
  
   // Fit scaler only on training data.
   scaleX.Fit(trainX);
   scaleX.Transform(trainX, scTrainX);
   scaleX.Transform(testX, scTestX);
  
   // Scale training predictions.
   // scaleY.Fit(trainY);
   // scaleY.Transform(trainY, scTrainY);
   // scaleY.Transform(testY, scTestY);
  
   // std::cout << trainX.submat(0, 0, trainX.n_rows - 1, 5).t() << '\n';
   // std::cout << scTrainX.submat(0, 0, trainX.n_rows - 1, 5).t() << '\n';
  
   //! - H1: The number of neurons in the 1st layer.
   constexpr int H1 = 64;
   //! - H2: The number of neurons in the 2nd layer.
   constexpr int H2 = 128;
   //! - H3: The number of neurons in the 3rd layer.
   constexpr int H3 = 64;
  
   // Number of epochs for training. Increase number of epochs for better results.
   const int EPOCHS = 30;
   // - STEP_SIZE: Step size of the optimizer.
   constexpr double STEP_SIZE = 5e-2;
   // - BATCH_SIZE: Number of data points in each iteration of SGD.
   constexpr int BATCH_SIZE = 32;
   // - STOP_TOLERANCE: Stop tolerance;
   // A very small number implies that we do all iterations.
   constexpr double STOP_TOLERANCE = 1e-8;
  
   // This intermediate layer is needed for connection between input data and the next LeakyReLU layer.
  
   // Standard Feed Forward Neural Network
  
   // OutputLayerType    Rule used to initialize the weight matrix.
   FFN<MeanSquaredError, HeInitialization> model;
   model.Add<Linear>(H1);
   // Activation layer:
   model.Add<LeakyReLU>();
   // Connection layer between two activation layers.
   model.Add<Linear>(H2);
   // Activation layer.
   model.Add<LeakyReLU>();
   // Connection layer.
   model.Add<Linear>(H3);
   // Activation layer.
   model.Add<LeakyReLU>();
   // Connection layer => output.
   // The output of one neuron is the regression output for one record.
   model.Add<Linear>(1);
  
   // Optimizer
   ens::Adam optimizer(
       STEP_SIZE,  
       BATCH_SIZE, 
       0.9,        // Exponential decay rate for the first moment estimates.
       0.999,      // Exponential decay rate for the weighted infinity norm estimates.
       1e-8,       // Value used to initialise the mean squared gradient parameter.
       trainData.n_cols * EPOCHS, // Max number of iterations.
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
  
   arma::mat nnPredictions;
   model.Predict(scTestX, nnPredictions);
   arma::rowvec nnPredY = round(nnPredictions);  // Set threshold
   
   ClassificationReport(nnPredY, testY);

  return 0;
}



