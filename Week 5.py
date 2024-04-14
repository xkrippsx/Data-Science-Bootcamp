import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso

# Load the dataset
data = pd.read_csv("employee.csv")

# Preprocessing
# Assuming the same preprocessing steps as in the lecture
# Split features and target
X = data.drop(columns=["salary"])
y = data["salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Get predictions on test data
test_predictions = linear_reg.predict(X_test_scaled)

# Compute Mean Absolute Error (MAE) and Mean Square Error (MSE) for test data
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)

print("Mean Absolute Error (MAE) on test data:", mae)
print("Mean Squared Error (MSE) on test data:", mse)

# Optional: Implement Ridge and Lasso Regression and compute evaluation metrics
ridge_reg = Ridge(alpha=1.0)  # Using default alpha value for now
ridge_reg.fit(X_train_scaled, y_train)
ridge_test_predictions = ridge_reg.predict(X_test_scaled)
ridge_mae = mean_absolute_error(y_test, ridge_test_predictions)
ridge_mse = mean_squared_error(y_test, ridge_test_predictions)
print("Ridge Regression:")
print("Mean Absolute Error (MAE) on test data:", ridge_mae)
print("Mean Squared Error (MSE) on test data:", ridge_mse)

lasso_reg = Lasso(alpha=1.0)  # Using default alpha value for now
lasso_reg.fit(X_train_scaled, y_train)
lasso_test_predictions = lasso_reg.predict(X_test_scaled)
lasso_mae = mean_absolute_error(y_test, lasso_test_predictions)
lasso_mse = mean_squared_error(y_test, lasso_test_predictions)
print("Lasso Regression:")
print("Mean Absolute Error (MAE) on test data:", lasso_mae)
print("Mean Squared Error (MSE) on test data:", lasso_mse)
