#include <mlpack.hpp>
#include "crow_all.h"
#include <iostream>
#include "generator/ModelGenerator.h"
#include "eval/ModelEvaluator.h"
#include "deserializer/PredictRequestDeserializer.h"

using namespace mlpack;

int main() {
 
  LinearRegression lr;
  DecisionTree<> dt;
  FFN<MeanSquaredError, RandomInitialization> nn;
  
  data::Load("models/lr.bin", "lr", lr);
  data::Load("models/dt.bin", "dt", dt);

  data::MinMaxScaler scalar;
  data::Load("data/scalar.bin", "scalar", scalar);

  arma::mat dataset;  
  data::DatasetInfo info;
  data::Load("data/cleaned_credit_data.csv", dataset, info);
  
  ModelGenerator modelGenerator(dataset); 

  // Index represents the Dimension. 
  // E.g "Senior Citizen" is in dimension 1. "Dependents" is in dimension 3
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
  
  PredictRequestDeserializer deserializer(info, dimensionToDataField);

  arma::mat dataX = dataset.submat(0, 0, dataset.n_rows - 2, dataset.n_cols - 1);
  arma::mat dataY = dataset.row(dataset.n_rows - 1);
  arma::mat scaledX;
  scalar.Transform(dataX, scaledX);

  crow::SimpleApp app;

  CROW_ROUTE(app, "/")([](){
    return "Customer Credit Risk Modelling";
  });

  CROW_ROUTE(app, "/generate")([&modelGenerator](){
    modelGenerator.generateBaseLinReg();
    modelGenerator.generateBaseDT();

    return "Models generated!";
  });


  CROW_ROUTE(app, "/linear/stats")([&](){
    std::string eval = ModelEvaluator::Eval(lr, dataX, dataY); 
    return crow::response(200, eval);
  });

  CROW_ROUTE(app, "/nn/stats")([&](){
    std::string eval = ModelEvaluator::Eval(nn, scaledX, dataY); 
    return crow::response(200, eval);
  });

  CROW_ROUTE(app, "/linear/predict").methods(crow::HTTPMethod::POST)
  ([&](const crow::request &req){
      auto body = crow::json::load(req.body);
      if (!body) 
        return crow::response(400, "Invalid body");
      arma::colvec input(19);
      
      try {
        deserializer.convertRequestBodyToInput(body, input);
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
        deserializer.convertRequestBodyToInput(body, input);
      } catch (const std::runtime_error &err) {
        return crow::response(400, "Invalid body");
      }

      arma::colvec scaledInput;
      scalar.Transform(input, scaledInput);

      arma::rowvec predictions;
      nn.Predict(scaledInput, predictions);
      std::ostringstream response;
      response << "Predictions: " << predictions << '\n';

      return crow::response(200, response.str());
  });


  app.port(3000).multithreaded().run();
  
}



