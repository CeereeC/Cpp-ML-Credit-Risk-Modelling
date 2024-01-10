# Cpp-ML-Credit-Risk-Modelling

## About

In this exercise, I build Linear Regression, Decision Tree and Forward Neural Network models using [mlpack](https://github.com/mlpack/mlpack), a fast, header-only C++ machine learning library. I then expose the models' functions via a REST API using [Crow](https://github.com/CrowCpp/Crow), a C++ Web Services Framework.

The task is to predict the likelihood of a customer defaulting on telco payments based on their telco data.
The customer dataset I used contains information about a fictional telco company that provides home phone and Internet services to 7048 customers. It indicates which customers have left, stayed, or signed up for their service.

## Tools required

C++14 compiler (For mlpack)  
mlpack >= 4.2.0 

## Getting Started
1. Clone the repository
2. Run the makefile to build all files

To just run the application, 
1. Go to [Releases](https://github.com/CeereeC/Cpp-ML-Credit-Risk-Modelling/releases)
2. Download ml-app.o
3. Run the command
   ```
   ./ml-app.o
   ```
## Interacting with the API

### 1. Model Prediction 
```
POST /lr/predict       // Use the linear regression model
POST /dt/predict       // Use the decision tree model
POST /nn/predict       // Use the neural network model
```
Post a json object of customer data. Returns the model prediction.

Sample Data:
```
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 58.0,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Two year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 55.5,
  "TotalCharges": 1421
}
```
Response:
```
Predictions:    0.2546
```
### 2. Model Metrics 
```
GET /lr/stats
GET /dt/stats 
GET /nn/stats      
```
Returns the metrics about the model.  
Response:
```
     precision         recall       f1-score        support

0        0.83            0.91          0.87         4.7e+03
1        0.67             0.5          0.57         9.4e+02
```
### 3. Regenerate Models 
```
GET /generate     
```
Generates the models, dataset encoders and scalars and saves them to disk.
Response:  
```
Models generated!
```
### 4. Load Models 
```
GET /load     
```
Loads the previously generated models, dataset encoders and scalars into memory.  
Response:  
```
Models loaded!
```
