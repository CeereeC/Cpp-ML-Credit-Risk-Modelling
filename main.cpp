#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include "crow_all.h"
#include <iostream>
#include "ModelGenerator.h"
#include "ModelEvaluator.h"

using namespace mlpack;

int main() {

  LinearRegression lr;
  data::Load("models/lr.bin", "lr", lr);

  FFN<MeanSquaredError, RandomInitialization> nn;;
  data::Load("models/nn.bin", "nn", nn);

  data::MinMaxScaler scalar;
  data::Load("data/scalar.bin", "scalar", scalar);

  data::DatasetInfo info;
  data::Load("data/dataset_info.bin", "dataset_info", info);

  arma::mat dataset;  
  data::Load("data/cleaned_credit_data.csv", dataset, info);

  arma::mat dataX = dataset.submat(0, 0, dataset.n_rows - 2, dataset.n_cols - 1);
  arma::mat dataY = dataset.row(dataset.n_rows - 1);
  arma::mat scaledX;
  scalar.Transform(dataX, scaledX);

  std::vector<std::string> dimensionToDataField = {
    "gender",
    "SeniorCitizen",
    "Partner",           
    "Dependents",        
    "tenure",            
    "PhoneService",      
    "MultipleLines",     
    "InternetService",   
    "OnlineSecurity",    
    "OnlineBackup",      
    "DeviceProtection",  
    "TechSupport",       
    "StreamingTV",       
    "StreamingMovies",   
    "Contract",         
    "PaperlessBilling",  
    "PaymentMethod",     
    "MonthlyCharges",    
    "TotalCharges"};

  crow::SimpleApp app;

  CROW_ROUTE(app, "/")([](){
    return "Customer Credit Risk Modelling";
  });

  CROW_ROUTE(app, "/linear/stats")([&](){
    std::string eval = ModelEvaluator::Evaluate(lr, dataX, dataY); 
    return crow::response(200, eval);
  });

  CROW_ROUTE(app, "/nn/stats")([&](){
    std::string eval = ModelEvaluator::Evaluate(nn, scaledX, dataY); 
    return crow::response(200, eval);
  });

  CROW_ROUTE(app, "/linear/predict").methods(crow::HTTPMethod::POST)
  ([&](const crow::request &req){
      auto body = crow::json::load(req.body);
      if (!body) 
        return crow::response(400, "Invalid body");
      arma::colvec input(19);
      
      try {
        for (size_t i = 0; i < 19; ++i) {
          auto field = body[dimensionToDataField[i]];
            std::string ss;
            if (field.t() == crow::json::type::String) {
              ss = field.s();   // If you don't do this, it'll return "abc" with the quotes
            } else {
              std::ostringstream os;
              os << field;
              ss = os.str();
            }
            input(i) = info.MapString<double>(ss, i);
        } 
      } catch (const std::runtime_error &err) {
        return crow::response(400, "Invalid body");
      }

      arma::rowvec predictions;
      lr.Predict(input, predictions);
      std::ostringstream response;
      response << "Predictions: " << predictions << '\n';

      return crow::response(200, response.str());
  });

    CROW_ROUTE(app, "/nn/predict").methods(crow::HTTPMethod::POST)
  ([&](const crow::request &req){
      auto body = crow::json::load(req.body);
      if (!body) 
        return crow::response(400, "Invalid body");
      arma::colvec input(19);
      
      try {
        for (size_t i = 0; i < 19; ++i) {
          auto field = body[dimensionToDataField[i]];
            std::string ss;
            if (field.t() == crow::json::type::String) {
              ss = field.s();   // If you don't do this, it'll return "abc" with the quotes
            } else {
              std::ostringstream os;
              os << field;
              ss = os.str();
            }
            input(i) = info.MapString<double>(ss, i);
        } 
      } catch (const std::runtime_error &err) {
        return crow::response(400, "Invalid body");
      }

      arma::colvec scaledInput;
      scalar.Transform(input, scaledInput);

      arma::rowvec predictions;
      nn.Predict(scaledInput, predictions);
      // std::cout << "Input: " << scaledInput << '\n';
      std::ostringstream response;
      response << "Predictions: " << predictions << '\n';

      return crow::response(200, response.str());
  });


  app.port(3000).multithreaded().run();
  
}



