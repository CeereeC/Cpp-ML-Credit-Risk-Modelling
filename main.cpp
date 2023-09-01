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

   // ============ Linear Regression ============= //
  
   // Regress.
   LinearRegression lr(trainX, trainY);
  
   // This will store the predictions; one row for each point.
   arma::rowvec predictions;
   lr.Predict(testX, predictions);   // Predict.
   arma::rowvec predY = round(predictions);  // Set threshold
  
   // Test
   ClassificationReport(predY, testY);

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
  std::cout << bestLambda << '\n';

  return 0;
}



