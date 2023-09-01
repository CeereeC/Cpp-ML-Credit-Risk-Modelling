#include <iostream>
#include <mlpack.hpp>

using namespace mlpack;

int main() {

  // Testing data is taken from the dataset in this ratio.
  constexpr double RATIO = 0.1; //10%

  arma::mat dataset;  
  data::DatasetInfo info;
  data::Load("data/cleaned_credit_data.csv", dataset, info); // Remember that Load(...) transposes the matrix

  return 0;
}



