# Stock Price Prediction using Machine Learning
![image](https://github.com/TANISHABASU26/Stock-Price-Prediction-using-Machine-Learning/assets/174117644/27e92082-9a9b-485f-86be-28cba203fee6)

This project demonstrates how to predict stock prices using machine learning models such as Support Vector Machine (SVM) and Linear Regression. The data used is from Quandl's WIKI dataset, specifically the adjusted close prices of Facebook (FB) stock.

# Key Business Metric Question:

Can machine learning models accurately predict the future stock prices of Facebook?

## Installation

Before you begin, ensure you have the required dependencies installed. You can install them using:

```bash
pip install quandl
pip install numpy
pip install scikit-learn
```

## Data Collection
We use the quandl package to fetch historical stock data. Specifically, we retrieve the adjusted close prices of Facebook (FB) stock:
```bash
import quandl

df = quandl.get("WIKI/FB")
```
## Data Preparation
The dataset is prepared by creating a new column, Prediction, which contains the adjusted close price shifted 30 days into the future. This allows us to predict stock prices 30 days ahead.
```bash
# Create another column (the target or dependent variable) shifted 'n' units up
forecast_out = 30
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
```
## Model Training and Testing
Support Vector Machine (SVM)
We create and train the SVM model with the radial basis function (RBF) kernel:
 ```bash
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
svr_rbf = SVR(C=1000.0, gamma=0.1)
svr_rbf.fit(X_train, y_train)

# Test the model
svm_confidence = svr_rbf.score(X_test, y_test)
print("SVM confidence:", svm_confidence)
```
## Linear Regression
We also create and train the Linear Regression model:
```bash
from sklearn.linear_model import LinearRegression

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Test the model
lr_confidence = lr.score(X_test, y_test)
print("Linear Regression confidence:", lr_confidence)
```
## Predictions
We predict the stock prices for the next 30 days using both models:
```bash
# Predict with Linear Regression
lr_prediction = lr.predict(x_forecast)
print("Linear Regression predictions:", lr_prediction)

# Predict with SVM
svm_prediction = svr_rbf.predict(x_forecast)
print("SVM predictions:", svm_prediction)
```
## Model Results
SVM Model: The SVM model achieved a confidence score of 0.9851.
Linear Regression Model: The Linear Regression model achieved a confidence score of 0.9805.

##Predictions
Linear Regression Predictions:
```bash
[177.11569728 183.55495033 183.9997339  181.37146735 180.00679049
 181.92744681 183.0191883  187.36593683 189.02376649 185.51604152
 182.34190423 179.93602946 180.62342225 184.44451747 183.81777698
 187.79050296 186.40560866 189.32702802 188.8519183  185.94060766
 188.2757214  187.94213372 189.18550597 176.51928295 172.06133853
 173.3148195  168.76589663 163.206102   163.88338608 155.9278254 ]
```
SVM Predictions:

```bash
[177.05157605 178.83179966 178.53818742 177.38946419 176.9998477
 177.8311302  178.92515673 187.06405471 179.14358128 181.88197534
 178.33404026 176.93401073 177.26924399 178.51295842 178.65523244
 185.09508247 186.66894972 178.9093843  179.55984417 184.31670514
 182.21713809 184.2141307  178.93971557 178.34531184 171.69847266
 172.21736357 172.18432145 168.03225907 166.14545708 157.83496617]
```
 # Outcomes
SVM Model: The SVM model's confidence score indicates a high level of accuracy in predicting future stock prices.
Linear Regression Model: The Linear Regression model also shows a high confidence score, suggesting reliable predictions.

#Critical Insights
Both models provide predictions with high confidence scores, but the SVM model slightly outperforms the Linear Regression model.
Accurate stock price prediction can significantly impact investment strategies and financial planning.

#Moving Forward Recommendations
Model Improvement: Consider using additional features (e.g., trading volume, other technical indicators) to improve model accuracy.
Model Comparison: Evaluate other machine learning models (e.g., Random Forest, Neural Networks) for better performance.
Real-time Predictions: Integrate real-time data for continuous model training and prediction updates.




