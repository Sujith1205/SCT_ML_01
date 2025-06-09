import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load training data
train_df = pd.read_csv(r'/Users/sujith/Desktop/skillcraft/train.csv')

# Select relevant features
features = train_df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
target = train_df['SalePrice']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

# Load test data
test_df = pd.read_csv(r'/Users/sujith/Desktop/skillcraft/test.csv')
test_features = test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]

# Predict prices for test data
test_predictions = model.predict(test_features)

# Save predictions to CSV
submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

output_path = r'/Users/sujith/Desktop/skillcraft/linear_regression_submission.csv'
submission_df.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")