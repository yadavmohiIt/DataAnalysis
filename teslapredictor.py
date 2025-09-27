import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Loaddata
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.fillna(method='ffill', inplace=True)

# Featureengineering
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df.dropna(inplace=True)

# Train-testsplit
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# TrainXGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predictandevaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Plotresults
plt.figure(figsize=(10, 5))
plt.plot(df['Date'].iloc[-len(y_test):], y_test, label='Actual', color='blue')
plt.plot(df['Date'].iloc[-len(y_test):], predictions, label='Predicted', color='orange')
plt.title("Tesla Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
