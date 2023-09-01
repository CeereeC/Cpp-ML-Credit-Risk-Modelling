#include <iostream>
#include <mlpack.hpp>

using namespace mlpack;

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

  return 0;
}



